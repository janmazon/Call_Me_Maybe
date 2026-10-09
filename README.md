*This project has been created as part of the 42 curriculum by jcamarer.*
 
# call me maybe
 
## Description
 
This project builds a strong "Function Calling" system using a Large Language Model (LLM). The main goal is to take structured information from natural language prompts and force the model to return a JSON object that is 100% valid and can always be parsed, following predefined schemas. To do this, I do not rely on prompt engineering. Instead, I use an advanced technique at the architecture level called Constrained Decoding.
 
## Instructions
 
The project uses `uv` as the package manager and a `Makefile` to automate the main tasks.
 
Installation:
 
```bash
make install
```
 
Run:
 
```bash
make run
```
 
Static analysis and cleaning (Linting & Clean):
 
```bash
make lint
make lint-strict
make clean
```
 
## Example Usage
 
You can run the program with custom paths for the input and output files using command line arguments:
 
```bash
uv run python -m src --output data/my_output/results.json
```
 
## Algorithm Explanation
 
The core of the system is Constrained Decoding. Unlike traditional Prompt Engineering, this algorithm works at a low level, directly inside the mathematical loop that generates text. The process works in these steps:
 
1. **Encoding and Context (ChatML):** The process starts by formatting the user input and the JSON schema definitions with the standard ChatML format (`<|im_start|>`, `<|im_end|>`). This helps the LLM separate the strict system instructions from the user's input text.
2. **Function Selection (Function Routing):** Before building the JSON, the system forces the model to decide which function to call. To do this, it blocks every token that does not help to form the exact name of one of the available functions. The model cannot invent names. It is mathematically "cornered" into choosing the predefined name that best fits the user's request.
3. **Speculative Evaluation (Real-time validation):** Once the function is selected, the JSON of the parameters starts to be generated, one token at a time. At each step, the program looks at the clean vocabulary of the model, takes the JSON text generated so far, adds each possible token to it, and passes this "proposed" text through a set of Regular Expression (Regex) validators.
4. **Strict Schema Enforcement:** The regular expressions check the JSON syntax (opening braces, double quotes for keys, separating commas) and are adapted to the data type that the function needs. If the function asks for a boolean parameter, the Regex mask only allows tokens that form the words `true` or `false`. If it asks for a number, the validator requires the correct decimal format and blocks letters (except 'e' or 'E', to allow scientific notation in edge cases).
5. **Logit Masking:** At each step, the language model returns a raw vector of probabilities (logits) for all the tokens in its vocabulary. The system takes this vector and applies a mask: if a token did not pass the validation test of the previous step, its mathematical value is artificially set to `-inf` (minus infinity).
6. **Final Token Selection:** Finally, using the `np.argmax` function, the system selects the token that the LLM originally thought was the most probable, but only choosing among the allowed tokens. The winning token is added to the context, which guarantees a perfect format, and the loop repeats until the closing brace `}` is safely generated.
## Design Decisions
 
- **Pydantic for Validation:** I used Pydantic in `models.py` to validate the input files when they are loaded. This makes sure that any format error in the definitions or in the tests stops the program in a safe way.
- **Safe Vocabulary (Clean Vocab):** I made an initial filter on the model's vocabulary to remove emojis, non-ASCII characters and text in unsupported languages. This reduces the number of mathematical steps and avoids decoding errors.
- **Robustness over Completeness (Edge Case Handling):** In extreme edge cases, the loop has a `max_tokens` limit. If the model cannot close a valid JSON before this limit, the program stops that prompt and returns an empty dictionary `{}`. This guarantees that the program never crashes, avoids infinite loops (timeouts), and always creates a final file that can be parsed.
## Performance Analysis
 
- **Speed:** By setting a token limit (`max_tokens = 50`) and using a safe brute-force approach, the program processes the tests in less than 5 minutes on standard hardware, which meets the time limit of the project.
- **Accuracy:** The system gets 100% accuracy on the public tests provided, and it successfully handles prompt injection attacks, SQL injections, and complex extractions of numbers and strings.
- **Reliability:** The rate of valid JSON generated is 100%. The tests show that the system does not fail or stop unexpectedly with malformed inputs.
## Challenges Faced
 
- **Number Validation with Regex:** One of the biggest challenges was to design regular expressions that check partial numbers token by token, but also force a strict floating-point (float) format and scientific notation, without false positives, before allowing the value to be closed. I solved it by creating dual conditions (partial match vs. full match) in the numbers block.
- **Closing Strings:** Stopping special characters inside an input string from breaking the closing quote of the JSON was an important challenge. I improved the regular expression step by step to allow escaped characters (`\\`) and any content inside the strings until they are closed.
## Testing Strategy
 
I designed a set of tests focused on common weak points:
 
- **File handling:** Files that do not exist, directories passed as files, corrupted JSONs, and missing write permissions. All are caught with clear exceptions.
- **Injected prompts:** SQL attacks (`; DROP TABLE...`) and contradictory instructions (`Ignore all instructions`).
- **Edge data types:** Mathematical precision checks with decimals, negative values, large numbers, scientific notation, and strings with line breaks and special characters.
## Resources
 
References:
 
- Official Python documentation (`json`, `re`, `argparse`, `typing`).
- Pydantic documentation.
Use of Artificial Intelligence:
 
I used an Artificial Intelligence Assistant (LLM) during the development of this project for the following purposes:
 
- **Concept tutoring:** To understand Constrained Decoding and the handling of Logits in depth.
- **Solving Regex problems:** To debug, structure and improve the complex regular expressions that do the mathematical and syntax validation token by token.