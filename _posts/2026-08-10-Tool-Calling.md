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
Once model returns these results its responsibility of our code to make the tool can and feed the results back to the model. Let's see how it works.

We have alredy seen that model returns the `tool_calls` back with the details of function name and argument details. So, step 1 is we just simply calls that tool with the provided argument.
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

After this we pass this back to Model to get final formatted natural sentenced answer. For example you might have asked two questions, so this way you get one answer from tool and one from model. So, our steps are:
1. Ask model our questions by passing tools.
2. Model returns list of tools.
3. We call tools manually.
4. We collect original quesion, model response, tool results and pass everything back to the model.
5. Then model gives us back the final answer.

```python
# Step 1: As the question.
question = "What is 47 plus 89? and how is weather in Tokyo?"

response = client.chat(
    model="qwen2.5:7b",
    messages=[{"role": "user", "content": question}],
    tools=tools
)

# Step 2: Model's first respons include tools (if there are any)
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
So, far we have seen a single tool call, but whatever we have learned so far can be easily modified to support multiple tool calls. To support this let's add anotehr simple tool `add_numbers`.

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

Now if we repeat the same thing as what we did in the section "Closing the Loop" by just modifying tool calling loop with the following
```python
# Execute each tool call and append its result
for tool_call in message.tool_calls:
    name = tool_call.function.name
    args = tool_call.function.arguments
    
    if name == "get_weather":
        result = get_weather(args["city"])
    elif name == "add_numbers":    # << Here we have addded our 2nd tool >>
            result = add_numbers(args["a"], args["b"])
    
    messages.append({"role": "tool", "content": result})

```
We get a slightly different output:
```
The sum of 47 and 89 is 136. 

Currently, the weather in Tokyo is 22°C and sunny.
```

Note: It doesn't say anymore that "For your math question....". Becasue both results are not coming back from our tool calls.


So, what we saw here is a Model can:
- determine when to use a tool.
- determine which tool to use.
- determine when not to use a tool.
- also use a tool along with its own answer.
- multiple tool calls

These all looks very simple and fun. But there comes scenarios where models fails to call a tool or calls an incorrect tools. We will see all that in one of the upcoming article. That is why it is very important to define tool/function defination clearly and define multiple tools with minimum ambiguty. 