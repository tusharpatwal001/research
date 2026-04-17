import streamlit as st
import requests
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="Product News Finder", page_icon="📰", layout="wide")

st.title("📰 Product News Finder with SearXNG")
st.markdown(
    "Enter products → get the **latest news** from SearXNG → displayed in a clean DataFrame."
)

# ====================== SIDEBAR CONFIG ======================
st.sidebar.header("⚙️ Settings")

searxng_url = st.sidebar.text_input(
    "SearXNG Instance URL",
    value="http://localhost:8080/",
    help="Paste any public instance from https://searx.space/",
)

time_range = st.sidebar.selectbox(
    "Latest news time range",
    options=["", "day", "week", "month", "year"],
    index=2,  # default = month
    help="Blank = no filter (most recent first)",
)

num_results = st.sidebar.slider(
    "News items per product",
    min_value=1,
    max_value=10,
    value=5,
    help="How many news articles to fetch per product",
)

st.sidebar.markdown("---")
st.sidebar.info("Public instances have rate limits. Keep the number of products small.")

# ====================== MAIN INPUT ======================
st.subheader("Enter your products")
products_input = st.text_area(
    "One product per line (e.g. iPhone 17, Tesla Cybertruck, Sony WF-1000XM6)",
    height=150,
    placeholder="iPhone 17\nSamsung Galaxy S25\nSony WH-1000XM6\n...",
)

if st.button("🔍 Search Latest News", type="primary", use_container_width=True):
    if not products_input.strip():
        st.warning("Please enter at least one product.")
    elif not searxng_url.strip().endswith("/"):
        st.error("SearXNG URL must end with a slash (/)")
    else:
        products = [p.strip() for p in products_input.split("\n") if p.strip()]

        st.info(f"Searching news for **{len(products)}** product(s) on SearXNG...")

        all_news = []
        progress_bar = st.progress(0)

        base_url = searxng_url.rstrip("/") + "/search"

        for i, product in enumerate(products):
            params = {"q": product, "categories": "news", "format": "json", "pageno": 1}
            if time_range:
                params["time_range"] = time_range

            try:
                with st.spinner(f"Fetching news for **{product}**..."):
                    resp = requests.get(base_url, params=params, timeout=15)
                    resp.raise_for_status()
                    data = resp.json()

                    results = data.get("results", [])[:num_results]

                    for r in results:
                        # Robust field extraction (different engines return slightly different keys)
                        published = (
                            r.get("publishedDate")
                            or r.get("date")
                            or r.get("published_date")
                            or r.get("age")
                            or "N/A"
                        )

                        snippet = (
                            r.get("content")
                            or r.get("snippet")
                            or "No snippet available"
                        )
                        source = r.get("source") or r.get("engine") or "Unknown"

                        all_news.append(
                            {
                                "Product": product,
                                "Title": r.get("title", "No title"),
                                "URL": r.get("url", "#"),
                                "Snippet": (
                                    snippet[:280] + "..."
                                    if len(snippet) > 280
                                    else snippet
                                ),
                                "Published": published,
                                "Source": source,
                            }
                        )

            except Exception as e:
                st.error(f"Failed to fetch news for **{product}**: {e}")

            # Update progress
            progress_bar.progress((i + 1) / len(products))

        progress_bar.empty()

        if all_news:
            df = pd.DataFrame(all_news)

            # Make URL column clickable
            st.dataframe(
                df,
                column_config={
                    "URL": st.column_config.LinkColumn(
                        "News Link",
                        help="Click to open the full article",
                        display_text="Open Article →",
                    ),
                    "Snippet": st.column_config.TextColumn("Snippet", width="large"),
                    "Published": st.column_config.TextColumn(
                        "Published", width="medium"
                    ),
                },
                use_container_width=True,
                hide_index=True,
            )

            # Optional download button
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name=f"product_news_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
            )

            st.success(
                f"✅ Found {len(all_news)} news articles across {len(products)} products!"
            )
        else:
            st.warning(
                "No news found. Try a different time range or check the SearXNG instance."
            )

st.caption(
    "Built for you with ❤️ using Streamlit + SearXNG. Public instances may have occasional rate limits."
)
