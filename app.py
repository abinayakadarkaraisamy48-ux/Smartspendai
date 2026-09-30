from pathlib import Path
import re

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="SmartSpend AI", page_icon="💰", layout="wide")

EXPECTED_COLUMNS = [
    "transaction_id", "payer_name", "payee_name", "date", "time",
    "amount", "category", "payment_mode", "city", "transaction_status"
]

ROOT = Path(__file__).resolve().parent
DATA_CANDIDATES = [
    ROOT / "SmartSpendAI_cleaned_dataset.csv",
    ROOT / "data" / "SmartSpendAI_cleaned_dataset.csv",
    ROOT / "SmartSpendAI_cleaned_dataset-1.csv",
    ROOT / "data" / "SmartSpendAI_cleaned_dataset-1.csv",
]

@st.cache_data
def load_dataset():
    for path in DATA_CANDIDATES:
        if path.is_file():
            return pd.read_csv(path)
    tried = "\\n".join(str(p) for p in DATA_CANDIDATES)
    raise FileNotFoundError("CSV dataset not found. Checked these paths:\\n" + tried)

def validate_dataset(data):
    return [c for c in EXPECTED_COLUMNS if c not in data.columns]

def preprocess_data(data):
    data = data.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["amount"] = pd.to_numeric(data["amount"], errors="coerce")
    data["category"] = data["category"].astype("string").str.strip()
    data = data.dropna(subset=["date", "amount", "category"])
    data = data[data["category"] != ""]
    for col in ["payer_name", "payee_name", "payment_mode", "city", "transaction_status"]:
        data[col] = data[col].fillna("Unknown").astype(str)
    data["time"] = data["time"].fillna("12:00").astype(str).str.strip()
    return data.reset_index(drop=True)

def create_features(data):
    data = data.copy()
    data["month"] = data["date"].dt.month
    data["month_name"] = data["date"].dt.month_name()
    data["day"] = data["date"].dt.day
    data["day_of_week"] = data["date"].dt.dayofweek
    data["day_name"] = data["date"].dt.day_name()
    parsed = pd.to_datetime(data["time"], format="%H:%M", errors="coerce")
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(data.loc[missing, "time"], format="%H:%M:%S", errors="coerce")
    data["hour"] = parsed.dt.hour.fillna(12).astype(int)
    data["weekend"] = (data["day_of_week"] >= 5).astype(int)
    data["spending_level"] = pd.cut(
        data["amount"], [-np.inf, 200, 500, 1000, 3000, np.inf],
        labels=["Very Low", "Low", "Medium", "High", "Very High"]
    )
    return data

def train_model(data):
    features = ["amount", "month", "day", "day_of_week", "hour", "weekend"]
    work = data.dropna(subset=features + ["category"]).copy()
    if len(work) < 5 or work["category"].nunique() < 2:
        return None
    X = work[features]
    encoder = LabelEncoder()
    y = encoder.fit_transform(work["category"].astype(str))
    counts = pd.Series(y).value_counts()
    stratify = y if counts.min() >= 2 else None
    test_size = max(1, int(round(len(work) * 0.2)))
    if stratify is not None:
        test_size = max(test_size, len(counts))
        if len(work) - test_size < len(counts):
            stratify = None
            test_size = max(1, int(round(len(work) * 0.2)))
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=stratify
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    model = RandomForestClassifier(
        n_estimators=150, random_state=42, class_weight="balanced", n_jobs=-1
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    labels = sorted(set(y_test) | set(pred))
    names = encoder.inverse_transform(labels)
    return {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, pred, average="weighted", zero_division=0),
        "matrix": confusion_matrix(y_test, pred, labels=labels),
        "names": names,
        "report": classification_report(y_test, pred, labels=labels, target_names=names, zero_division=0),
    }

def extract_sms(sms):
    text = str(sms).lower()
    match = re.search(r"(?:rs\\.?|inr|₹)\\s*([\\d,]+(?:\\.\\d{1,2})?)", str(sms), re.I)
    amount = float(match.group(1).replace(",", "")) if match else None
    payee = "Unknown"
    for pattern in [r"\\bto\\s+([A-Za-z0-9 &._-]+)", r"\\bat\\s+([A-Za-z0-9 &._-]+)", r"\\bfor\\s+([A-Za-z0-9 &._-]+)"]:
        found = re.search(pattern, str(sms), re.I)
        if found:
            payee = found.group(1).strip(" .,-")
            break
    rules = {
        "Food": ["swiggy", "zomato", "restaurant", "food", "cafe"],
        "Transport": ["uber", "ola", "bus", "train", "metro", "travel"],
        "Shopping": ["amazon", "flipkart", "myntra", "shopping"],
        "Entertainment": ["movie", "netflix", "spotify", "entertainment"],
        "Education": ["college", "book", "education", "course"],
        "Health": ["hospital", "clinic", "pharmacy", "medicine"],
        "Bills": ["bill", "electricity", "water", "internet"],
    }
    category = "Others"
    for label, words in rules.items():
        if any(word in text for word in words):
            category = label
            break
    return {"amount": amount, "payee": payee, "category": category}

def get_insights(data):
    if data.empty:
        return ["No valid transaction data is available."]
    total = float(data["amount"].sum())
    by_category = data.groupby("category")["amount"].sum().sort_values(ascending=False)
    top = str(by_category.index[0])
    top_amount = float(by_category.iloc[0])
    insights = [
        f"Total spending recorded: ₹{total:,.2f}.",
        f"Average transaction: ₹{data['amount'].mean():,.2f}.",
        f"Highest transaction: ₹{data['amount'].max():,.2f}.",
        f"Highest spending category: {top} (₹{top_amount:,.2f}, {top_amount / total * 100 if total else 0:.1f}% of total).",
    ]
    weekend = data.loc[data["weekend"] == 1, "amount"].sum()
    weekday = data.loc[data["weekend"] == 0, "amount"].sum()
    if weekend > weekday:
        insights.append("Recorded weekend spending is higher than weekday spending.")
    elif weekday > weekend:
        insights.append("Recorded weekday spending is higher than weekend spending.")
    else:
        insights.append("Recorded weekend and weekday spending are equal.")
    return insights

def get_recommendations(data):
    if data.empty or data["amount"].sum() <= 0:
        return ["Add valid transactions with a positive total to generate recommendations."]
    total = float(data["amount"].sum())
    sums = data.groupby("category")["amount"].sum().sort_values(ascending=False)
    top = str(sums.index[0])
    recs = [f"Review your {top} spending; it represents {float(sums.iloc[0]) / total * 100:.1f}% of recorded spending."]
    if "Food" in sums.index and float(sums["Food"]) / total > 0.15:
        recs.append("Consider setting a weekly food budget that suits your needs.")
    if "Shopping" in sums.index:
        recs.append("Review non-essential shopping purchases against your budget.")
    if "Entertainment" in sums.index:
        recs.append("Consider setting a monthly entertainment budget.")
    recs.append("Choose a realistic savings target after accounting for essential expenses.")
    return recs

st.markdown("<h1>💰 SmartSpend AI</h1>", unsafe_allow_html=True)
st.caption("Intelligent Student Expense Analytics & Savings Assistant")

try:
    raw_df = load_dataset()
except Exception as exc:
    st.error("Dataset could not be loaded.")
    st.code(str(exc))
    st.info("Put the CSV beside app.py or inside the data/ folder. The filename may be SmartSpendAI_cleaned_dataset.csv.")
    st.stop()

missing = validate_dataset(raw_df)
if missing:
    st.error("Dataset is missing required columns:")
    st.write(missing)
    st.stop()

df = preprocess_data(raw_df)
if df.empty:
    st.error("No valid rows remain after cleaning. Check date, amount, and category values.")
    st.stop()
feature_df = create_features(df)

page = st.sidebar.radio("Navigation", [
    "🏠 Home", "📊 Dashboard", "📥 Transactions", "🧹 Data Cleaning",
    "📈 EDA", "🔧 Feature Engineering", "🤖 Machine Learning",
    "💡 AI Insights", "🎯 Saving Planner", "🔬 Data Science Process", "🔐 Privacy"
])

if page == "🏠 Home":
    st.subheader("Project Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions", f"{len(df):,}")
    c2.metric("Total Spending", f"₹{df['amount'].sum():,.2f}")
    c3.metric("Categories", int(df["category"].nunique()))
    c4.metric("Cities", int(df["city"].nunique()))
    st.write("SmartSpend AI analyzes student transactions, summarizes spending patterns, and offers budgeting suggestions.")
    st.subheader("Workflow")
    st.write("Data → Validation → Cleaning → EDA → Feature Engineering → Random Forest → Evaluation → Insights → Savings Planner")

elif page == "📊 Dashboard":
    st.subheader("Expense Dashboard")
    a, b, c, d = st.columns(4)
    a.metric("Total Spending", f"₹{df['amount'].sum():,.2f}")
    b.metric("Average Transaction", f"₹{df['amount'].mean():,.2f}")
    c.metric("Largest Transaction", f"₹{df['amount'].max():,.2f}")
    d.metric("Successful Transactions", f"{df['transaction_status'].astype(str).str.strip().str.lower().eq('success').mean() * 100:.1f}%")
    left, right = st.columns(2)
    with left:
        st.write("Spending by Category")
        st.bar_chart(df.groupby("category")["amount"].sum().sort_values(ascending=False))
    with right:
        st.write("Transactions by Payment Mode")
        st.bar_chart(df["payment_mode"].value_counts())
    st.write("Monthly Spending")
    monthly = df.groupby(df["date"].dt.to_period("M"))["amount"].sum()
    monthly.index = monthly.index.astype(str)
    st.line_chart(monthly)
    st.write("City-wise Spending")
    st.bar_chart(df.groupby("city")["amount"].sum().sort_values(ascending=False))
    st.write("Transaction Status")
    st.bar_chart(df["transaction_status"].value_counts())

elif page == "📥 Transactions":
    tab1, tab2, tab3 = st.tabs(["Dataset", "Add Transaction", "SMS Extraction"])
    with tab1:
        st.dataframe(df, use_container_width=True)
        st.download_button("Download cleaned CSV", df.to_csv(index=False).encode("utf-8"), "SmartSpendAI_cleaned_dataset.csv", "text/csv")
        if "added_df" in st.session_state:
            st.download_button("Download CSV with added transaction", st.session_state["added_df"].to_csv(index=False).encode("utf-8"), "SmartSpendAI_updated_dataset.csv", "text/csv")
    with tab2:
        with st.form("transaction_form"):
            c1, c2 = st.columns(2)
            payer = c1.text_input("Payer Name", "Student")
            payee = c2.text_input("Payee Name")
            tx_date = c1.date_input("Date")
            tx_time = c2.time_input("Time")
            amount = c1.number_input("Amount (₹)", min_value=0.0, step=10.0)
            categories = sorted(df["category"].astype(str).unique().tolist())
            if "Others" not in categories:
                categories.append("Others")
            category = c2.selectbox("Category", categories)
            modes = sorted(df["payment_mode"].astype(str).unique().tolist()) or ["UPI", "Cash"]
            mode = c1.selectbox("Payment Mode", modes)
            city = c2.text_input("City", "Unknown")
            status = st.selectbox("Status", ["Success", "Failed", "Pending"])
            submit = st.form_submit_button("Add transaction")
        if submit:
            row = {
                "transaction_id": f"MANUAL-{pd.Timestamp.now().strftime('%Y%m%d%H%M%S%f')}",
                "payer_name": payer or "Student", "payee_name": payee or "Unknown",
                "date": str(tx_date), "time": tx_time.strftime("%H:%M"), "amount": amount,
                "category": category, "payment_mode": mode, "city": city or "Unknown",
                "transaction_status": status,
            }
            st.session_state["added_df"] = pd.concat([raw_df, pd.DataFrame([row])], ignore_index=True)
            st.success("Added for this session. Download the updated CSV from the Dataset tab.")
    with tab3:
        sms = st.text_area("Paste bank/payment SMS")
        if st.button("Extract SMS"):
            result = extract_sms(sms)
            st.json(result)
            if result["amount"] is None:
                st.warning("Amount not detected. Include a value like Rs. 250 or ₹250.")

elif page == "🧹 Data Cleaning":
    st.subheader("Raw Data Preview")
    st.dataframe(raw_df.head(20), use_container_width=True)
    st.subheader("Missing Values")
    st.dataframe(raw_df.isna().sum().rename("Missing values"))
    st.write(f"Rows before cleaning: {len(raw_df):,}")
    st.write(f"Rows after cleaning: {len(df):,}")
    st.write(f"Rows removed: {len(raw_df) - len(df):,}")
    st.dataframe(df.head(20), use_container_width=True)
    st.download_button("Download cleaned data", df.to_csv(index=False).encode("utf-8"), "SmartSpendAI_cleaned_dataset.csv", "text/csv")

elif page == "📈 EDA":
    st.subheader("Exploratory Data Analysis")
    left, right = st.columns(2)
    with left:
        st.write("Transaction Amount Distribution")
        hist = pd.cut(df["amount"], bins=10, duplicates="drop").value_counts().sort_index()
        hist.index = hist.index.astype(str)
        st.bar_chart(hist)
    with right:
        st.write("Transactions by Category")
        st.bar_chart(df["category"].value_counts())
    st.write("Category Summary")
    st.dataframe(df.groupby("category")["amount"].agg(["count", "sum", "mean"]).sort_values("sum", ascending=False))

elif page == "🔧 Feature Engineering":
    st.write("Created calendar, hour, weekend, and spending-level features.")
    st.dataframe(feature_df.head(100), use_container_width=True)
    st.download_button("Download feature CSV", feature_df.to_csv(index=False).encode("utf-8"), "SmartSpendAI_features.csv", "text/csv")

elif page == "🤖 Machine Learning":
    result = train_model(feature_df)
    if result is None:
        st.warning("Not enough data/categories for a reliable train/test split. Add more records across at least two categories.")
    else:
        a, b, c, d = st.columns(4)
        a.metric("Accuracy", f"{result['accuracy']:.3f}")
        b.metric("Precision", f"{result['precision']:.3f}")
        c.metric("Recall", f"{result['recall']:.3f}")
        d.metric("F1 Score", f"{result['f1']:.3f}")
        st.subheader("Classification Report")
        st.code(result["report"])
        st.subheader("Confusion Matrix")
        st.dataframe(pd.DataFrame(result["matrix"], index=result["names"], columns=result["names"]))

elif page == "💡 AI Insights":
    for insight in get_insights(feature_df):
        st.info(insight)

elif page == "🎯 Saving Planner":
    budget = st.number_input("Monthly budget (₹)", min_value=0.0, value=10000.0, step=500.0)
    goal = st.number_input("Monthly savings goal (₹)", min_value=0.0, value=2000.0, step=500.0)
    spent = float(df["amount"].sum())
    a, b, c = st.columns(3)
    a.metric("Recorded spending (all rows)", f"₹{spent:,.2f}")
    b.metric("Budget remaining", f"₹{budget - spent:,.2f}")
    c.metric("Savings goal", f"₹{goal:,.2f}")
    st.warning("This comparison uses all valid rows in the dataset, not only the current month.") if spent > budget else st.success("Recorded spending is within the entered budget.")
    for recommendation in get_recommendations(feature_df):
        st.write("• " + recommendation)

elif page == "🔬 Data Science Process":
    steps = [
        ("Data Collection", "Load transaction records from CSV."),
        ("Validation", "Check required columns."),
        ("Cleaning", "Convert dates and amounts; remove invalid essential rows."),
        ("EDA", "Summarize category, time, city, mode, and status."),
        ("Feature Engineering", "Create calendar and time-based features."),
        ("Model Training", "Train a Random Forest classifier."),
        ("Evaluation", "Display accuracy, precision, recall, F1, and confusion matrix."),
        ("Insights", "Summarize spending patterns."),
        ("Saving Planner", "Compare recorded spending with a user-entered budget."),
    ]
    for title, description in steps:
        st.subheader(title)
        st.write(description)

elif page == "🔐 Privacy":
    st.write("Do not commit real bank messages, account numbers, or sensitive personal data to a public repository.")
    st.write("SMS extraction is rule-based. Check the extracted details before relying on them.")
