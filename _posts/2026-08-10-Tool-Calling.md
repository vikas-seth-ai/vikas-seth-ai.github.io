---
layout: post
title:  "Tool Calling"
date:   2026-08-09 19:17:50 -0400
categories: tool-calling ollama ai-models llm
---

# Function Calling with Local LLMs: A Working Walkthrough

## 1. Why this write-up
Tool calling lets a language model request that a function be executed on its behalf, rather than trying to compute or fabricate an answer itself. I wanted to actually build the full loop — not just read about it — using a local model running through Ollama. This covers the mechanics: how a tool call is requested, how you feed results back, and where structured output fits alongside it.


## 2. The basic mechanics of a tool call
Let's see what a tool looks like, what we send to the model, and what comes back.


```python
import ollama
# Use the URL of your Ollama machine
client = ollama.Client(host="http://192.168.5.168:11434")
```

Our first tool — let's create a Get Weather tool, which the model will call whenever we ask for weather information.
```python
def get_weather(city: str) -> str:
    # This is a placeholder implementation. 
    # In a real scenario, you would call a weather API to get the actual weather data.
    return f"The weather in {city} is 22°C and sunny."
```
But how do we let our model know that we have this tool available?
We create a tools array with the details of our function, which explains its type (function), name of the function (get_weather), description, parameters, and properties of each parameter. This specification is read by the model while determining which tool to use.

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
]
```

### Passing Tools while calling model
Here is a simple example of how we can include a tool while calling a model:
```python
response = client.chat(
    model="qwen2.5:7b",
    messages=[{"role": "user", "content": "How is current weather in Berlin?"}],
    tools=tools
)
print(response["message"])
```

Along with the prompt, we are also passing the `tools` parameter. What it returns is a *request* to call a specific tool with specific arguments.
```
role='assistant' content='' thinking=None images=None tool_name=None tool_calls=[ToolCall(function=Function(name='get_weather', arguments={'city': 'Berlin'}))]
```
It returns which tool to call — in this case `get_weather` — and what arguments to pass, `city = Berlin`.

Let's see a couple of variations.
#### What happens when we don't pass `tools`
```python
response = client.chat(
    model="qwen2.5:7b",
    messages=[{"role": "user", "content": "How is current weather in Berlin?"}],
)
print(response["message"])
```

**Output**: It returns a regular response from the model.
```
role='assistant' content="I don't have real-time access to current weather conditions. As of my last update, I can suggest checking a reliable weather website or app for the most accurate and up-to-date information on the current weather in Berlin. Common sources include the Weather Channel, BBC Weather, or local German meteorological services." thinking=None images=None tool_name=None tool_calls=None
```

#### What if we ask a non-weather-related question and still pass `tools`?
```python
response = client.chat(
    model="qwen2.5:7b",
    messages=[{"role": "user", "content": "what is 47 plus 89?"}],
    tools=tools
)
print(response["message"])
```

**Output**: It gives us the answer, and tells us that no tools were used for this.

```
role='assistant' content='The sum of 47 and 89 is 136.' thinking=None images=None tool_name=None tool_calls=None
```



## 3. Closing the loop — feeding results back
Once the model returns these results, it's our code's responsibility to call the tool and feed the results back to the model. Let's see how it works.

We've already seen that the model returns `tool_calls` with the function name and argument details. So step 1 is simply to call that tool with the provided arguments.
Here is a sample code for it:
```python
for tool_call in response["message"]["tool_calls"]:
    tool_name = tool_call["function"]["name"]
    tool_args = tool_call["function"]["arguments"]
    if tool_name == "get_weather":
        city = tool_args.get("city")
        if city:
            results = get_weather(city)
```

After this, we pass it back to the model to get the final, naturally formatted answer. For example, you might have asked two questions, so this way you get one answer from the tool and one from the model. So, our steps are:
1. Ask the model our question, passing the tools.
2. Model returns the list of tool calls.
3. We call the tools manually.
4. We collect the original question, the model's response, and the tool results, and pass everything back to the model.
5. Then the model gives us back the final answer.

```python
# Step 1: Ask the question.
question = "What is 47 plus 89? and how is weather in Tokyo?"

response = client.chat(
    model="qwen2.5:7b",
    messages=[{"role": "user", "content": question}],
    tools=tools
)

# Step 2: Model's first response includes tools (if there are any)
message = response["message"]

# Build up the conversation history as we go (Part of Step 4)
messages = [
    {"role": "user", "content": question},
    {"role": "assistant", "content": "", "tool_calls": message.tool_calls}
]

# Step 3: Execute each tool call and append its result
for tool_call in message.tool_calls:
    name = tool_call.function.name
    args = tool_call.function.arguments
    
    if name == "get_weather":
        result = get_weather(args["city"])

    # Here we can add logic for any additional tools we might have.
    
    messages.append({"role": "tool", "content": result}) # Part of Step 4

# Step 5: Now ask for the final natural-language answer
follow_up = client.chat(
    model="qwen2.5:7b",
    messages=messages,
    tools=tools
)

print(follow_up["message"]["content"])
```

Model Response:
```
The weather in Tokyo is currently 22°C and sunny.

For your math question, 47 plus 89 equals 136.
```


## 4. Multiple tool calls in one turn
So far, we've seen a single tool call, but everything we've learned so far can easily be extended to support multiple tool calls. To support this, let's add another simple tool, `add_numbers`.

```python
def add_numbers(a: float, b: float) -> str:
    return str(a + b)
```
We will need to modify our `tools` list to include this tool as well

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
    },
    {
        "type": "function",
        "function": {
            "name": "add_numbers",
            "description": "Add two numbers together and return the sum",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "The first number"},
                    "b": {"type": "number", "description": "The second number"}
                },
                "required": ["a", "b"]
            }
        }
    }
]
```

Now, if we repeat what we did in the "Closing the Loop" section, just modifying the tool-calling loop as follows:
```python
# Execute each tool call and append its result
for tool_call in message.tool_calls:
    name = tool_call.function.name
    args = tool_call.function.arguments
    
    if name == "get_weather":
        result = get_weather(args["city"])
    elif name == "add_numbers":    # << Here we've added our 2nd tool >>
            result = add_numbers(args["a"], args["b"])
    
    messages.append({"role": "tool", "content": result})

```
We get a slightly different output:
```
The sum of 47 and 89 is 136. 

Currently, the weather in Tokyo is 22°C and sunny.
```

Note: It no longer says "For your math question…". Because both results are now coming back from our tool calls.


So what we saw here is that a model can:
- determine when to use a tool.
- determine which tool to use.
- determine when not to use a tool.
- also use a tool along with its own answer.
- multiple tool calls

This all looks very simple and fun, but there are scenarios where models fail to call a tool or call an incorrect tool. We'll see all of that in an upcoming article. That's why it's very important to define tool/function definitions clearly, and define multiple tools with minimal ambiguity.