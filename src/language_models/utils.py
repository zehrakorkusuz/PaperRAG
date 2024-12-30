import time
import json

def log_metrics(eval_count, eval_duration):
    """
    Log performance metrics.

    :param eval_count: Number of tokens generated.
    :param eval_duration: Duration of evaluation in nanoseconds.
    """
    tokens_per_second = (eval_count / eval_duration) * 1e9
    print(f"Tokens Generated: {eval_count}")
    print(f"Duration (s): {eval_duration / 1e9:.2f}")
    print(f"Speed: {tokens_per_second:.2f} tokens/second")

def stream_responses(language_model, messages, schema=None, json_format=True, log_metrics_fn=None):
    """
    Stream responses from a language model with optional structured output.
    
    :param language_model: Instance of a language model.
    :param messages: List of messages to send to the model.
    :param schema: Optional JSON schema for structured output.
    :param log_metrics_fn: Optional logging function to log performance metrics.
    """
    print("Assistant: ", end='', flush=True)
    start_time = time.time_ns() 
    eval_count = 0

    request_params = {"messages": messages}
    if schema or json_format:
        request_params["format"] = "json"

    accumulated_content = ""  

    for response in language_model.send_request(**request_params, stream=True):
        try:
            # Parse the response line as JSON
            response_data = json.loads(response)
            content = response_data.get("message", {}).get("content", "")
            done = response_data.get("done", False)

            # Print the content as it streams, formatted with indentation
            if content.strip():
                # Check if content is a valid JSON, and if so, format it
                try:
                    formatted_content = json.loads(content)
                    print(json.dumps(formatted_content, indent=2), end='', flush=True)  # Proper indentation
                except json.JSONDecodeError:
                    # If it's not JSON, print it as plain text
                    print(content, end='', flush=True)
                
                eval_count += len(content.split())  # Count tokens based on whitespace

            # Accumulate content for final structured parsing
            accumulated_content += content

            # Check if the response is done, exit the loop if it is / last item is metadata
            if done:
                print("\nResponse complete.")
                break

        except (json.JSONDecodeError, TypeError) as e:
            print(f"\nError: Could not parse response as JSON. Raw response: {response}")
            print(f"Exception: {str(e)}")

    if accumulated_content.strip():
        try:
            if schema:
                # Attempt to parse the accumulated content with the schema
                structured_content = json.loads(accumulated_content)
                print("\nStructured Output:")
                print(json.dumps(structured_content, indent=2), flush=True)  # Format with newlines
            else:
                # If no schema, just print the accumulated content
                print("\nFinal Content:")
                print(accumulated_content, flush=True)
            eval_count += len(accumulated_content.split())  # Add token count for the final part
        except json.JSONDecodeError as e:
            print(f"\nError: Could not parse the accumulated content as JSON. Exception: {str(e)}")

    # End timing and log performance metrics if needed
    end_time = time.time_ns()  # End timing
    eval_duration = end_time - start_time

    if log_metrics_fn:
        log_metrics_fn(eval_count, eval_duration)




