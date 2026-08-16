---
layout: post
title:  "Tool Calling"
date:   2026-08-09 19:17:50 -0400
categories: jekyll update
---

# Function Calling with Local LLMs: A Working Walkthrough

## 1. Why this write-up
Tool calling lets a language model request that a function be executed on its behalf, rather than trying to compute or fabricate an answer itself. I wanted to actually build the full loop — not just read about it — using a local model running through Ollama. This covers the mechanics: how a tool call is requested, how you feed results back, and where structured output fits alongside it.

Data points to reference:
- Model: `qwen2.5:7b` for tool calling, `qwen2.5:14b-instruct` for structured output extraction, both via local Ollama.
- Two toy tools used throughout: `get_weather(city)` and `add_numbers(a, b)`

## 2. The basic mechanics of a tool call
Let's see how a tool looks like and what do we send to the model and what comes back.


```python
import ollama
# Use the URL of your Ollama machine
client = ollama.Client(host="http://192.168.5.168:11434")
```

Our first tool - Let's create a Get Weather tool which model will call whenever we ask for weather information.
```python
def get_weather(city: str) -> str:
    # This is a placeholder implementation. 
    # In a real scenario, you would call a weather API to get the actual weather data.
    return f"The weather in {city} is 22°C and sunny."
```
But how do we let our model know that we have this tool available.
We create a tools array, with the details of our function. Which also explains it type (function), name of the function (get_weather), description, parameters and properties of paramter. This specification is read my the model while determining which tool to use.

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a given city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "The city name"}
                },
                "required": ["city"]
            }
        }
    }
```

### Passing Tools while calling model
Here is a simple example how can we include a tool while calling a model
```python
response = client.chat(
    model="qwen2.5:7b",
    messages=[{"role": "user", "content": "How is current weather in Berlin?"}],
    tools=tools
)
print(response["message"])
```

Here along with the prompt we are also passing `tools` paramter. And what it returns is a *request* to call a specific tool with specific arguments.
```
role='assistant' content='' thinking=None images=None tool_name=None tool_calls=[ToolCall(function=Function(name='get_weather', arguments={'city': 'Berlin'}))]
```
It returns which tool to call, like in this case `get_weather`, what arguments to pass `city = Berlin`.

Let's see a couple of variations.
#### What happens when we don't pass `tools`
```python
response = client.chat(
    model="qwen2.5:7b",
    messages=[{"role": "user", "content": "How is current weather in Berlin?"}],
)
print(response["message"])
```

**Output**: It returns a regular output comging back from the model.
```
role='assistant' content="I don't have real-time access to current weather conditions. As of my last update, I can suggest checking a reliable weather website or app for the most accurate and up-to-date information on the current weather in Berlin. Common sources include the Weather Channel, BBC Weather, or local German meteorological services." thinking=None images=None tool_name=None tool_calls=None
```

#### What if we ask not weather related question and still pass `tools`.
```python
response = client.chat(
    model="qwen2.5:7b",
    messages=[{"role": "user", "content": "what is 47 plus 89?"}],
    tools=tools
)
print(response["message"])
```

**Output**: It gives us the answer, and tells us that no tools are available for this.

```
role='assistant' content='The sum of 47 and 89 is 136.' thinking=None images=None tool_name=None tool_calls=None
```



## 3. Closing the loop — feeding results back
Once model returns these results its responsibility of our code to make the tool can and feed the results back to the model. Let's see how it work

Data to include:
- After calling `get_weather("Tokyo")` yourself and getting back "The weather in Tokyo is 22°C and sunny," that result was appended to the conversation as a `tool` role message
- Model then produced a natural sentence combining the result: confirms the loop is genuinely two round-trips, not one

## 4. Correctly declining to use a tool
[Vikas: this is an easy thing to get wrong when building agents — the model needs to recognize when *no* tool applies.]

Data to include:
- Prompt: "Where is Tokyo?" with both tools still available → model answered directly in text, no tool call, even though `get_weather` existed and mentioned Tokyo
- Prompt: "What's the capital of France?" → same, answered directly, ignored irrelevant tools
- Point worth making: having tools available doesn't force their use — the model is weighing relevance, at least in these clear-cut cases

## 5. Multiple tool calls in one turn
[Vikas: cover the parallel-call case.]

Data to include:
- Prompt: "What is 47 plus 89? and how is weather in Tokyo?" → model returned *two* tool calls in a single response (`add_numbers` and `get_weather`), not one at a time
- Your code has to loop through all returned tool calls, execute each, and append all results before asking for the final answer — a detail that's easy to miss if you only handle the single-tool-call case

## 6. Structured output — a different but related capability
[Vikas: distinguish this from tool calling — no "calling" happens here, the model is just constrained to emit JSON matching a schema.]

Data to include:
- Used `qwen2.5:14b-instruct` with a JSON schema (name, age, occupation) and `format=schema` — model returned syntactically valid JSON on the first attempt
- But: syntactically valid isn't the same as *correct*. On one run, `occupation` came back with a stray smart-quote character and trailing artifacts (`"software engineer", '`) — parsed as valid JSON structurally but wrong content
- This is why validation needs to check content, not just "did `json.loads()` succeed"

## 7. Building a validate-and-retry wrapper
[Vikas: describe the pattern, not just the code — why content-level checks matter.]

Data to include:
- Wrapper checked: name is alphabetic and non-empty, occupation is alphanumeric/whitespace only, age is an integer in a sane range (0-120)
- On a real run, attempt 1 failed content validation (malformed occupation string) even though JSON parsing succeeded; attempt 2 passed
- Takeaway: a validation layer that only checks "is this valid JSON" will silently let through structurally-fine-but-wrong data — worth stating this as a general principle, not just specific to this test

## 8. What this sets up
[Vikas: 2-3 sentences bridging to Write-up #2 — you now had a working loop and a validation habit, which is what let you notice something was wrong when the failure-mode testing produced empty responses.]

---
### Appendix: reference summary (for your writing, cut or keep in final piece)

| Capability tested | Result |
|---|---|
| Single tool call, correct routing | ✅ Correct tool + arguments |
| No-tool-needed recognition | ✅ Declined tool use twice |
| Parallel multi-tool calls | ✅ Two tool calls in one response |
| Full execute-and-respond loop | ✅ Final answer combined both results |
| Structured output (schema-constrained) | ⚠️ Valid JSON, but content occasionally malformed |
| Retry-with-content-validation | ✅ Caught and recovered from bad content |