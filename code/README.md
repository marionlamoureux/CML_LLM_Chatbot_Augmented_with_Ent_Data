# Project Build Process

The following step-by-step instructions correspond to the project files in this directory and should be followed in sequential order.

## Creating a Project
The first step for the second part of the lab will be to create another project within CML using another git repo as a starting point. For this you will need to:

 1. In a CML workspace, click **New Project**, add a Project Name (we recommend adding your user to avoid having duplicate names), and a description
 2. Select Git as the Initial Setup option and add the repo URL: https://github.com/nhernandezdlm/CML_LLM_Chatbot_Augmented_with_Enterprise_Data.git
 3. Select Runtime setup **Basic**, Python 3.9 kernel.
 4. Tick on the **Add GPU enabled Runtime variant**
 5. Click **Create Project**
 ![create_project](../images/create_project.png)


## 0_session-resource-validation
Before starting the project, a GPU resource validation needs to happen. 

For this you will need to create a session (this session will be used on the next step - installing the dependencies as well). Follow these steps:
1. Go to Sessions --> **New Session**
2. Select **Workbench** for the Editor and **Python 3.9** for the Kernel, and Edition **Nvidia GPU**
3. You must select 1 GPU from the **Resource Profile**. 
4. Create session
![new_session](../images/new_session.png)

Once the session is running,open the `check_gpu_capability.py` script, then click **Run > Run All**.
Repeteat the same process with the `check_gpu_resources.py` script. This will do the initial checks.


## 1_session-install-deps
To install the dependencies for this project, you will need to run the script `code/1_session-install-deps/install_dependencies.py` by opening the script and clicking **Run > Run All**. this will install the python dependencies specified in `code/1_session-install-deps/requirements.txt`

## 2_job-download-models
Definition of the job **Download Models** 
- Directly download specified models from huggingface repositories
- These are pulled to new directories models/llm-model and models/embedding-model

## 3_job-populate-vectordb
Definition of the job **Populate Vector DB with documents embeddings**
- Start the milvus vector database and set database to be persisted in new directory milvus-data/
- Generate embeddings for each document in data/
- The embeddings vector for each document is inserted into the vector database
- Stop the vector database

## 4_app
Definition of the application `CML LLM Chatbot`
- Start the milvus vector database using persisted database data in milvus-data/
- Load locally persisted pre-trained models from models/llm-model and models/embedding-model 
- Start gradio interface 
- The chat interface performs both retrieval-augmented LLM generation and regular LLM generation for bot responses.

