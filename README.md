# 🔐 Permission-Aware RAG

**An AI-powered document question-answering system that ensures users can access only the information they are authorized to see.**

## 💡 What is this project?

Imagine two different companies using the same AI assistant.

Both companies have their own documents, reports, and confidential information. They also have access to some public documents.

Now, if an employee from Company A asks the AI about Company B's private documents, should the AI provide that information?

Of course not!

This is the problem that **Permission-Aware RAG (Retrieval-Augmented Generation)** aims to solve.

It allows an AI assistant to answer questions using documents that a user is authorized to access, while preventing access to private information belonging to other organizations.

## 🎯 What Problem Does It Solve?

Organizations manage large numbers of documents, making it difficult to find information while maintaining proper access control.

A normal AI assistant may generate answers using information that should not be visible to every user.

This project adds an authorization layer to document-based AI question answering.

Before retrieving information, the system checks who is asking the question and retrieves only the documents that user is allowed to access.

## ⚙️ How Does It Work?

1. **User Identification:** The system identifies the user through their configured account.
2. **Permission Checking:** It checks which organization the user belongs to and their access level.
3. **Document Retrieval:** It searches for relevant information only within authorized documents.
4. **Access Control:** Public documents are available to everyone, while private documents remain restricted to their respective organizations.
5. **Answer Generation:** The AI generates an answer using the retrieved information.
6. **Source References:** Answers include references to the documents used.
7. **Insufficient Evidence:** If the available documents do not contain enough information, the system avoids making up an answer.

## 🔒 Example: How Access Control Works

The current prototype supports three types of users:

| User | Document Access |
|---|---|
| Public Analyst | Public documents |
| Tenant A Analyst | Public + Tenant A private documents |
| Tenant B Analyst | Public + Tenant B private documents |

**Example scenarios:**

- Tenant A asks about its own security policy → Answer provided.
- Tenant B asks about its own security policy → Answer provided.
- Tenant A asks about Tenant B's private security policy → Access denied.
- Tenant A asks about public information → Answer provided.

This demonstrates how private information can be separated between different organizations using the same AI platform.

## 🌍 Where Can It Be Used?

Permission-Aware RAG can be useful in several areas:

- **Companies:** Managing documents and confidential information across different organizations and teams.
- **Hospitals:** Controlling access to patient and administrative records.
- **Banks:** Protecting confidential customer and financial information.
- **Educational Institutions:** Managing student, faculty, and administrative documents.
- **Multi-Client Platforms:** Allowing multiple customers to use one AI platform without exposing each other's private information.

These are potential use cases. The current prototype demonstrates organization-level access control rather than complete production systems for these industries.

## 🧰 Technologies Used

- **Python** – Main programming language.
- **Retrieval-Augmented Generation (RAG)** – Retrieves relevant document information before generating answers.
- **ChromaDB** – Stores and retrieves document embeddings.
- **Sentence Transformers** – Converts text into numerical embeddings.
- **Ollama** – Runs the language model locally.
- **Gemma 3 (4B)** – Generates answers using the retrieved evidence.

## 📁 Project Structure

```text
permission-aware-rag/
│
├── data/
│   ├── public/
│   ├── processed/
│   ├── chunks/
│   ├── embeddings/
│   └── chroma_db/
│
├── src/
│   ├── ingest.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── chroma_setup.py
│   ├── relevance.py
│   └── retrieval.py
│   └── generation.py
│
├── tests/
│
├── users.json
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

Follow these steps to run the project locally.

### 1. Clone the Repository

```bash
git clone https://github.com/Aviraj516/permission-aware-rag.git
cd permission-aware-rag
```

### 2. Create a Virtual Environment

```bash
python3 -m venv .venv
```

Activate it:

**macOS / Linux:**

```bash
source .venv/bin/activate
```

**Windows:**

```bash
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Ollama

Install Ollama and download the Gemma 3 model:

```bash
ollama pull gemma3:4b
```

Make sure the Ollama service is running.

### 5. Configure User Identity

The prototype uses `users.json` to identify the development user.

**For Tenant A:**

```bash
export RAG_USER_ID=employee_a
```

**For Tenant B:**

```bash
export RAG_USER_ID=employee_b
```

**For Public Access:**

```bash
export RAG_USER_ID=public_analyst
```

For Windows PowerShell, use:

```powershell
$env:RAG_USER_ID="employee_a"
```

### 6. Configure the Language Model

```bash
export RAG_LLM_MODEL=gemma3:4b
```

Windows PowerShell:

```powershell
$env:RAG_LLM_MODEL="gemma3:4b"
```

### 7. Run the Project

Make sure the ChromaDB collection has been prepared and populated with the required documents.

Run:

```bash
python -m src.retrieval
```

You can now ask questions through the terminal.

Type `exit` to stop the application.

## 🧪 Testing

Run the test suite:

```bash
python -m pytest -v
```

The tests cover:

- Tenant-level document access restrictions.
- Cross-tenant access prevention.
- Public document accessibility.
- Retrieval relevance.
- Answer generation.
- Citation validation.
- Insufficient evidence handling.

## 🔐 Security Note

This is currently a learning and development prototype.

The `RAG_USER_ID` environment variable is used for local development identity selection. It is not production authentication.

A production-ready implementation would require proper login, identity verification, secure permission management, and additional security testing.

## 📌 Current Project Scope

- Public document access.
- Tenant A and Tenant B document separation.
- Permission-aware document retrieval.
- Cross-tenant access denial.
- AI-generated answers using retrieved evidence.
- Citation validation.
- Insufficient-evidence handling.
- Local language model execution.

**Note:** Department-level permissions within an individual tenant are not currently implemented.

## 🚧 Future Improvements

- Implement proper user authentication.
- Add department-level access permissions.
- Support document uploads and management.
- Improve retrieval accuracy and answer quality.
- Add a user-friendly interface.
- Expand support for different document formats.

## 👨‍💻 Project Status

This project is actively being developed and improved.

The goal is to explore how AI can make document search and question answering more useful while respecting privacy and access permissions.

**The idea is simple: AI should not only know how to answer a question. It should also know which information it is allowed to use.**

---

⭐ If you find this project interesting, feel free to explore the repository and share your feedback.

**GitHub Repository:** https://github.com/Aviraj516/permission-aware-rag