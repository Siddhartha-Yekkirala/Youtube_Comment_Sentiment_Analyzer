
# YouTube Comment Sentiment Analyzer 🎥💬

A Python-based tool that fetches comments from YouTube videos and analyzes their sentiment as **Positive**, **Negative**, or **Neutral** using Natural Language Processing (NLP).

This project helps content creators, marketers, and researchers quickly understand audience feedback without manually reading thousands of comments.

---

## 🚀 Features

- Fetch comments from any YouTube video using **YouTube Data API v3**
- Perform sentiment analysis on comments
- Classify comments as **Positive / Negative / Neutral**
- Display overall sentiment summary
- Simple and easy-to-use Python implementation

---

## 🛠️ Tech Stack

- **Language:** Python  
- **API:** YouTube Data API v3  
- **NLP:** VADER / TextBlob / Transformer-based models  
- **Libraries:** pandas, nltk, google-api-python-client  

---

## 📂 Project Structure
youtube-comment-sentiment-analyzer/
│
├── main.py
├── sentiment_analysis.py
├── requirements.txt
├── README.md
└── .env.example

---

## 🔑 Prerequisites

- Python 3.7+
- Google Cloud account
- YouTube Data API v3 key

---

## ⚙️ Installation & Setup

1️⃣ Clone the Repository
bash
git clone https://github.com/yesh6289/youtube-comment-sentiment-analyzer.git
cd youtube-comment-sentiment-analyzer

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Set Up YouTube API Key
	•	Go to Google Cloud Console
	•	Enable YouTube Data API v3
	•	Create an API key
	•	Add it to a .env file
YOUTUBE_API_KEY=your_api_key_here

▶️ How to Run
python main.py
Enter the YouTube video URL when prompted.

📊 Sample Output
Total Comments Analyzed: 250

Positive: 60%
Neutral: 25%
Negative: 15%




🧠 How It Works
	1.	Takes a YouTube video URL as input
	2.	Fetches comments using YouTube Data API
	3.	Cleans and preprocesses text
	4.	Applies sentiment analysis model
	5.	Displays sentiment summary




📌 Use Cases
	•	YouTube content analysis
	•	Audience feedback monitoring
	•	Social media sentiment analysis
	•	NLP learning project

	

🤝 Contributing

Contributions are welcome!
	1.	Fork the repository
	2.	Create a new branch
	3.	Commit your changes
	4.	Open a Pull Request




📄 License

This project is licensed under the MIT License.




⭐ Acknowledgements
	•	YouTube Data API
	•	Open-source NLP libraries
	•	Python community
