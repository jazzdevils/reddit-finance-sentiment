import praw
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
from datetime import datetime

# 1) Reddit 클라이언트 설정
reddit = praw.Reddit(
    client_id="YOUR_CLIENT_ID",
    client_secret="YOUR_CLIENT_SECRET",
    user_agent="tsla-sentiment by u/your_username",
)

analyzer = SentimentIntensityAnalyzer()

def fetch_tsla_reddit_sentiment(subreddit_name="stocks", limit=100):
    subreddit = reddit.subreddit(subreddit_name)
    rows = []

    # 2) TSLA 언급 글 검색
    for submission in subreddit.search("TSLA", sort="new", limit=limit):
        # 글 제목 감성
        vs_title = analyzer.polarity_scores(submission.title)
        rows.append({
            "type": "post",
            "subreddit": subreddit_name,
            "post_id": submission.id,
            "created": datetime.fromtimestamp(submission.created_utc),
            "text": submission.title,
            "compound": vs_title["compound"],
        })

        # 3) 댓글 감성
        submission.comments.replace_more(limit=0)
        for c in submission.comments.list():
            vs_c = analyzer.polarity_scores(c.body)
            rows.append({
                "type": "comment",
                "subreddit": subreddit_name,
                "post_id": submission.id,
                "created": datetime.fromtimestamp(c.created_utc),
                "text": c.body,
                "compound": vs_c["compound"],
            })

    return pd.DataFrame(rows)

df = fetch_tsla_reddit_sentiment("stocks", limit=50)

def label_sentiment(compound: float) -> str:
    if compound >= 0.05:
        return "pos"
    elif compound <= -0.05:
        return "neg"
    else:
        return "neu"

def aggregate_daily_tsla(df: pd.DataFrame):
    d = df.copy()
    d["date"] = d["created"].dt.date
    d["label"] = d["compound"].apply(label_sentiment)

    return d.groupby("date").agg(
        avg_compound=("compound", "mean"),
        pos_ratio=("label", lambda x: (x == "pos").mean()),
        neg_ratio=("label", lambda x: (x == "neg").mean()),
        count=("text", "count"),
    ).reset_index()

daily = aggregate_daily_tsla(df)
print(daily)