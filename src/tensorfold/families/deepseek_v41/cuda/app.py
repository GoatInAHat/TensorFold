"""OpenAI routes for the DeepSeek-V4.1 engine: TensorFold's CUDA App, with ignore_eos passed to the engine, image
input (--vision) laid out as DeepSeek's encoding lays it out, and every reasoning_effort name mapped to DeepSeek's
numeric effort."""

from __future__ import annotations

import dataclasses
from typing import Any, Callable

from tensorfold.cuda.server import App, PreparedRequest
from tensorfold.server.errors import RequestError


# reasoning_effort -> the "Reasoning Effort: N" (1-100) line DeepSeek's encoding puts before the first turn when
# thinking. DeepSeek names low / high / max (50 / 75 / 100); minimal, medium and xhigh sit between them, so every
# OpenAI tier reaches the model as its own value (none or off: thinking off). No effort sent: high, as the reference.
EFFORT_VALUES = {"minimal": 25, "low": 50, "medium": 62, "high": 75, "xhigh": 88, "max": 100}


# The grounding tokens: DeepSeek's encoding keeps them as tokens in an assistant's content and reasoning (a model wrote
# them there). Every other special-token spelling in message text is plain text.
GROUNDING = ("<｜ref｜>", "<｜/ref｜>", "<｜box｜>", "<｜/box｜>",
             "<｜point｜>", "<｜/point｜>", "<｜polygon｜>", "<｜/polygon｜>")


class DsApp(App):
    reads_ignore_eos = True
    plain_message_text = True           # DeepSeek's encoding: a special-token spelling in message text is plain text
    plain_assistant_keep = GROUNDING

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        registry = getattr(self.engine, "adapters", None)
        if registry is not None:                        # --adapter: each adapter is a model id of its own
            self.adapter_models = registry.model_map()
        # --vision-max-images: images take DsVision.split's path, not the shared image limits, so its own count
        # check gets the server's value (DsVision's default is 8)
        vision, images = getattr(self.engine, "vision", None), kwargs.get("vision_max_images")
        if vision is not None and images is not None:
            vision.max_images = int(images)
        template = self.template
        template.efforts = frozenset(EFFORT_VALUES)     # the server passes every name through unchanged
        # DeepSeek's encoding keeps a second leading system message (Codex's developer turn, say) as its own
        # <｜System｜> turn after the tools block: not merged into the first
        template.merge_leading = False
        render = template.render

        def render_effort(messages, *, enable_thinking: bool, extra: dict[str, Any] | None = None, **kw: Any) -> str:
            extra = dict(extra or {})
            if enable_thinking:
                name = extra.get("reasoning_effort") or "high"
                extra["reasoning_effort"] = EFFORT_VALUES.get(name, name)
            return render(messages, enable_thinking=enable_thinking, extra=extra, **kw)

        template.render = render_effort

    @staticmethod
    def _response_format(body: dict[str, Any], chat: bool) -> dict[str, Any]:
        """A chat's response_format (json_object or json_schema) also reaches the model as DeepSeek's encoding writes
        it: a "## Response Format" block in the system turn (the template's response_format input). The engine then
        enforces it with the request's grammar, as it enforces regex, choice and grammar (``constrain``)."""

        rf = body.get("response_format")
        if not chat or not isinstance(rf, dict) or rf.get("type") not in ("json_object", "json_schema"):
            return body
        kwargs = body.get("chat_template_kwargs")
        if kwargs is not None and not isinstance(kwargs, dict):
            raise RequestError("chat_template_kwargs must be a JSON object or null")
        kwargs = dict(kwargs or {})
        spec = rf.get("json_schema") if rf.get("type") == "json_schema" else None
        # json_schema: DeepSeek's API layer (deepseek-ai/deepseek-recipe) writes the schema, not the OpenAI wrapper
        shown = spec["schema"] if isinstance(spec, dict) and isinstance(spec.get("schema"), dict) else rf
        kwargs.setdefault("response_format", shown)     # a client's own template input stays
        return {**body, "chat_template_kwargs": kwargs}

    def run(self, body: dict[str, Any], chat: bool, emit: Callable[[dict[str, Any]], bool], *,
            prepared: PreparedRequest | None = None, cancelled: Callable[[], bool] | None = None) -> dict[str, Any]:
        self.engine.request.stop_eos = not bool(body.get("ignore_eos", False))
        return super().run(body, chat, emit, prepared=prepared, cancelled=cancelled)

    def _prepare(self, body: dict[str, Any], chat: bool) -> PreparedRequest:
        """A chat with image_url parts (user turns and tool results): each image becomes DeepSeek's placeholder
        block, the prompt renders as text, and each placeholder token grows into the image's span."""

        body = self._response_format(body, chat)
        vision = getattr(self.engine, "vision", None)
        messages = body.get("messages") if chat else None
        if vision is None or not isinstance(messages, list) or not vision.has_images(messages):
            return super()._prepare(body, chat)
        try:
            text_messages, pictures = vision.split(messages)
        except (ValueError, OSError) as exc:
            raise RequestError(f"image input: {exc}") from exc
        from .vision import PLACEHOLDER

        # split refused every placeholder that the client wrote, so each one left in content is the server's own
        prepared = super()._prepare({**body, "messages": text_messages}, chat, plain_keep=(PLACEHOLDER,))
        try:
            vp = vision.expand(prepared.prompt, pictures)
        except ValueError as exc:
            raise RequestError(f"image input: {exc}") from exc
        return dataclasses.replace(prepared, prompt=vp.token_ids, vision=vp,
                                   sampling=self.sampling_for(body, vp.token_ids))
