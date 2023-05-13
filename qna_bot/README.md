To set up the repository locally, follow these steps:

1. Clone the repository: Clone the repository that contains the code to your local machine. You can use Git to clone the repository or download it as a ZIP file and extract it.

2. Set up a virtual environment (optional but recommended): It's good practice to create a virtual environment to isolate the dependencies of this project. Open a terminal or command prompt, navigate to the project directory, and create a virtual environment by running the following command:

   ```bash
   python3 -m venv myenv
   ```

   This command creates a new virtual environment named `myenv`. You can replace `myenv` with the desired name for your virtual environment.

3. Activate the virtual environment: Activate the virtual environment using the appropriate command for your operating system:

   - For Windows:

     ```bash
     myenv\Scripts\activate
     ```

   - For macOS/Linux:

     ```bash
     source myenv/bin/activate
     ```

4. Install dependencies: Install the required dependencies by running the following command in the terminal or command prompt:

   ```bash
   pip install langchain openai chromadb tiktoken pypdf panel gradio Pillow
   ```

   This command installs the necessary Python packages for the code to run. Make sure you have an internet connection to download the packages.

5. Set the OpenAI API key: Replace the placeholder `'YOUR_OPENAI_API_KEY'` in the code with your actual OpenAI API key. If you don't have an API key, you can sign up for one on the OpenAI website.

   ```python
   os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"
   ```

6. Prepare the required data: Make sure you have the `sui.pdf` file available and copy the path of the file. Then, open the `sui_blockchain_qna_bot.py` file and locate the following line:

   ```python
   loader = PyPDFLoader('/content/sui.pdf')
   ```

   Replace `'/content/sui.pdf'` with the actual path of the `sui.pdf` file on your local machine. For example:

   ```python
   loader = PyPDFLoader('/path/to/sui.pdf')
   ```

   Make sure to provide the correct file path and ensure that the file exists at that location. If the file is not present, you need to provide the PDF file you want to use or update the code accordingly.

   Note: If you're unsure about the file path, you can place the `sui.pdf` file in the same directory as the `sui_blockchain_qna_bot.py` file and update the line to:

   ```python
   loader = PyPDFLoader('sui.pdf')
   ```

   This assumes that both files are in the same directory.

7. Run the code: Save the changes and run the code in the terminal or command prompt using the following command:

   ```bash
   python sui_blockchain_qna_bot.py
   ```

   The code will launch a Gradio interface for the QnA bot, allowing you to interact with it.

That's it! You have set up the repository locally for the QnA bot code. You can now use the bot by entering queries through the Gradio interface.
