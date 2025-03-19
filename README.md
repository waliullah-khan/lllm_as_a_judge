![Uploading image.png…]()

# 🤖 LLM as a Judge

*A Streamlit-based evaluation tool that compares chatbot responses generated by multiple Large Language Models (LLMs), including OpenAI GPT, Anthropic Claude, DeepSeek, and Perplexity.*

---

## 📖 Overview

The Multi-LLM Chatbot Evaluator helps you assess and compare the quality of chatbot responses from various LLM providers. It processes your chat history data, generates responses using selected LLMs, and evaluates them using semantic similarity, word-level F1 scores, and citation metrics.

---

## 🚀 Key Features

- **Multi-LLM Support:** Evaluate responses from multiple LLMs simultaneously:
  - OpenAI GPT (`gpt-3.5-turbo`)
  - Anthropic Claude (`claude-3-5-sonnet`)
  - DeepSeek Chat
  - Perplexity Sonar Small Online

- **Automated Metrics:** 
  - Semantic Similarity (cosine similarity of sentence embeddings)
  - Word-level F1 Score (precision and recall based on token overlap)
  - Citation Score (based on provided citations)

- **Interactive UI:** Built with Streamlit for intuitive interaction:
  - Upload chat history JSON files
  - Select desired LLMs for comparison
  - View detailed evaluation results in a structured table
  - Easily download evaluation results as CSV

---

## 🛠️ Tech Stack

| Component                 | Technology                             |
|---------------------------|----------------------------------------|
| Web Framework             | [Streamlit](https://streamlit.io/)     |
| Data Processing           | [Pandas](https://pandas.pydata.org/)   |
| Embeddings & Similarity   | [SentenceTransformers](https://www.sbert.net/) |
| LLM APIs                  | OpenAI, Anthropic, DeepSeek, Perplexity |
| Error Handling & Retries  | [Tenacity](https://tenacity.readthedocs.io/) |

---

## 🔑 Environment Variables

To run this project locally or deploy it, set the following environment variables in your `.env` file:

```bash
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
DEEPSEEK_API_KEY=your_deepseek_api_key
PERPLEXITY_API_KEY=your_perplexity_api_key
```

---

## ⚙️ Installation & Setup

### Step-by-step guide:

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/multi-llm-chatbot-evaluator.git
cd multi-llm-chatbot-evaluator
```

2. **Create and activate virtual environment**

```bash
python -m venv venv
source venv/bin/activate   # Linux/MacOS
.\venv\Scripts\activate    # Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Run the Streamlit app**

```bash
streamlit run main.py
```

---

## 📂 Usage

1. Open the Streamlit app in your browser.
2. Upload your chat history JSON file.
3. Select the LLMs you want to evaluate.
4. Click "Run Evaluation" to generate responses and calculate metrics.
5. Review the detailed results displayed in a structured table.
6. Download results as CSV for further analysis.

---

## 📊 Evaluation Metrics Explained

| Metric                | Description                                                | Ideal Range |
|-----------------------|------------------------------------------------------------|-------------|
| **Similarity**        | Cosine similarity between embeddings (semantic closeness). | ≥0.80       |
| **F1 Score**          | Word overlap precision & recall between responses.         | ≥0.75       |
| **Citation Score**    | Ratio of citations provided (out of max 5).                | ≥0.60       |

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

Please adhere to this project's `code of conduct`.

---

## ❓ FAQ

**Q: What file format should my chat history be?**  
A: JSON format with specific fields (`sender`, `content`, `assistThreadId`, etc.). See sample provided in the repository.

**Q: Can I add more LLM providers?**  
A: Yes! You can extend the app by adding new API integrations following existing patterns.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 📧 Contact

Wali Ullah Khan – [khan636@purdue.edu](mailto:khan636@purdue.edu)  
Project Link: [https://github.com/waliullah-khan/llm_as_a_judge](https://github.com/waliullah-khan/llm_as_a_judge)

---

## 🙌 Acknowledgements & Credits

Special thanks to these resources and libraries:

- [Streamlit Documentation](https://docs.streamlit.io/)
- [SentenceTransformers](https://www.sbert.net/)
- [Tenacity Library](https://tenacity.readthedocs.io/)
- API Providers: OpenAI, Anthropic, DeepSeek, Perplexity AI.

---

_Last updated: March 19, 2025_
