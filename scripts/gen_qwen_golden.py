#!/usr/bin/env python3
"""Render Qwen3.5 chat-template goldens with the checkpoint's own jinja.

tojson is overridden to compact json.dumps (ensure_ascii=False) to match Go's
json.Encoder with SetEscapeHTML(false) — HF's runtime uses json.dumps too
(no HTML escaping), so this is faithful for the checkpoint.
"""
import json, sys
import jinja2

SNAP = "/opt/spark/models/models--Qwen--Qwen3.5-35B-A3B-FP8/snapshots/0b2752837483aa34b3db6e83e151b150c0e00e49"
template_src = json.load(open(f"{SNAP}/tokenizer_config.json"))["chat_template"]

template_src = template_src.replace(
    "{{- '<|im_start|>' + message.role + '\\n' + content }}",
    "{{- '<|im_start|>' + message.role + '\\n<think>\\n\\n</think>\\n\\n' + content }}")

env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
env.filters["tojson"] = lambda v: json.dumps(v, ensure_ascii=False, separators=(",", ":"))
env.globals["raise_exception"] = lambda msg: (_ for _ in ()).throw(Exception(msg))
tmpl = env.from_string(template_src)

WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "days": {"type": "integer", "description": "Forecast days"},
                "metric": {"type": "boolean"},
            },
            "required": ["city"],
        },
    },
}
EDIT = {
    "type": "function",
    "function": {
        "name": "edit_file",
        "description": "Edit a file",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "edits": {"type": "array", "items": {"type": "string"}},
                "opts": {"type": "object"},
            },
            "required": ["path"],
        },
    },
}

cases = []

def case(name, messages, tools=None, thinking=True):
    out = tmpl.render(messages=messages, tools=tools, add_generation_prompt=True,
                      enable_thinking=thinking)
    cases.append({"name": name, "messages": messages, "tools": tools or [],
                  "thinking": thinking, "expected": out})

case("plain_think_on",
     [{"role": "system", "content": "You are helpful."},
      {"role": "user", "content": "Hi there"}])

case("plain_think_off",
     [{"role": "system", "content": "You are helpful."},
      {"role": "user", "content": "Hi there"}], thinking=False)

case("no_system",
     [{"role": "user", "content": "Just a question"}], thinking=False)

case("tools_system",
     [{"role": "system", "content": "Be terse."},
      {"role": "user", "content": "Weather in Paris?"}],
     tools=[WEATHER], thinking=False)

case("tools_no_system",
     [{"role": "user", "content": "Weather in Paris?"}],
     tools=[WEATHER, EDIT], thinking=True)

case("multiturn_assistant",
     [{"role": "system", "content": "Be terse."},
      {"role": "user", "content": "First question"},
      {"role": "assistant", "content": "First answer", "reasoning_content": "old thoughts"},
      {"role": "user", "content": "Second question"}],
     thinking=True)

case("tool_loop",
     [{"role": "user", "content": "Weather in Paris and Rome?"},
      {"role": "assistant", "content": "", "reasoning_content": "I should check both cities",
       "tool_calls": [
           {"type": "function", "function": {"name": "get_weather",
            "arguments": {"city": "Paris", "days": 3, "metric": True}}},
           {"type": "function", "function": {"name": "get_weather",
            "arguments": {"city": "Rome"}}},
       ]},
      {"role": "tool", "content": "Paris: 18C sunny"},
      {"role": "tool", "content": "Rome: 24C clear"},
      {"role": "user", "content": "And tomorrow?"}],
     tools=[WEATHER], thinking=True)

case("call_with_prose",
     [{"role": "user", "content": "Fix the bug"},
      {"role": "assistant", "content": "Let me look at the file.",
       "reasoning_content": "need to read it",
       "tool_calls": [
           {"type": "function", "function": {"name": "edit_file",
            "arguments": {"path": "a.go", "edits": ["x", "y"], "opts": {"dry": False}}}},
       ]},
      {"role": "tool", "content": "ok"}],
     tools=[EDIT], thinking=True)

json.dump(cases, open(sys.argv[1], "w"), ensure_ascii=False, indent=1)
print(f"wrote {len(cases)} cases")
