"""传统方法基线：TF-IDF（unigram+bigram）+ 朴素贝叶斯 / 逻辑回归

用法：
  python -m antifraud.baselines.tfidf_baseline \
      --train data/processed/train.jsonl --test data/processed/test.jsonl \
      --out results/tfidf_baseline.json
"""

from __future__ import annotations

import argparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from antifraud.data.dataset import load_split
from antifraud.training.metrics import macro_f1, per_class_prf
from antifraud.training.utils import read_jsonl, write_json


def evaluate(clf, X, y_true, label_names):
    y_pred = [int(p) for p in clf.predict(X)]
    return {
        "accuracy": float(sum(a == b for a, b in zip(y_true, y_pred)) / max(1, len(y_true))),
        "macro_f1": macro_f1(y_true, y_pred),
        "per_class": per_class_prf(y_true, y_pred, label_names),
        "predictions": y_pred,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="TF-IDF + NB / LR 基线")
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--out", default="results/tfidf_baseline.json")
    args = parser.parse_args()

    train_records = load_split(args.train)
    test_records = load_split(args.test)
    label_names = ["正常", "推销", "刷单", "钓鱼"]

    X_train_text = [r["text"] for r in train_records]
    y_train = [r["label"] for r in train_records]
    X_test_text = [r["text"] for r in test_records]
    y_test = [r["label"] for r in test_records]
    print(f"训练 {len(X_train_text)} 条 / 测试 {len(X_test_text)} 条")

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)
    print(f"TF-IDF 特征维度：{X_train.shape[1]}")

    results = {}
    for name, clf in {
        "tfidf_nb": MultinomialNB(),
        "tfidf_lr": LogisticRegression(max_iter=1000, n_jobs=-1),
    }.items():
        clf.fit(X_train, y_train)
        result = evaluate(clf, X_test, y_test, label_names)
        result.pop("predictions")
        results[name] = result
        print(f"{name}: accuracy={result['accuracy']:.4f} macro_f1={result['macro_f1']:.4f}")

    write_json(args.out, {"label_names": label_names, "n_train": len(y_train), "n_test": len(y_test), "results": results})
    print(f"结果已保存到 {args.out}")


if __name__ == "__main__":
    main()
