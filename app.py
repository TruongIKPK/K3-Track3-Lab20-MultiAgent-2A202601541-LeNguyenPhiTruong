"""Streamlit Web UI for Multi-Agent Research Lab & Benchmark Suite."""

import json
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd
import streamlit as st

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import AgentName, BenchmarkMetrics, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import (
    run_benchmark,
    run_corpus_benchmark,
    run_multi_agent_workflow,
    run_single_agent_baseline,
)
from multi_agent_research_lab.evaluation.report import render_markdown_report

st.set_page_config(
    page_title="Multi-Agent Research Lab",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for rich styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4F46E5, #06B6D4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .agent-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
        margin-right: 6px;
    }
    .agent-supervisor { background-color: #E0E7FF; color: #3730A3; }
    .agent-researcher { background-color: #FEF3C7; color: #92400E; }
    .agent-analyst { background-color: #EDE9FE; color: #5B21B6; }
    .agent-writer { background-color: #D1FAE5; color: #065F46; }
    .agent-critic { background-color: #FEE2E2; color: #991B1B; }
    </style>
    """,
    unsafe_allow_html=True,
)

CORPUS_DIR = Path(
    "src/ai_agent_offline_research_corpus_30_topics_v2/ai_agent_offline_research_corpus_v2/topics"
)


def load_corpus_topics() -> dict[str, Path]:
    """Load all topic JSON files."""
    if not CORPUS_DIR.exists():
        return {}
    topic_files = sorted(CORPUS_DIR.glob("*.json"))
    topics_map = {}
    for p in topic_files:
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            name = data.get("topic", {}).get("name", p.stem)
            topics_map[f"{data.get('benchmark_metadata', {}).get('topic_id', '')} - {name}"] = p
        except Exception:
            topics_map[p.stem] = p
    return topics_map


topics_map = load_corpus_topics()
settings = get_settings()

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/artificial-intelligence.png", width=64)
    st.title("System Settings")
    st.caption("Lab 20: Multi-Agent Systems")

    st.divider()
    st.markdown("**LLM Backend:** `OpenAI gpt-4o-mini`")
    st.markdown(
        f"**OpenAI API:** {'✅ Configured' if settings.openai_api_key else '⚠️ Using Mock Fallback'}"
    )
    st.markdown(
        f"**Langfuse Tracing:** {'✅ Active' if settings.langfuse_secret_key else '⚠️ Inactive'}"
    )

    if settings.langfuse_secret_key:
        st.link_button("🌐 Open Langfuse Dashboard", "https://cloud.langfuse.com")

    st.divider()
    st.markdown("### Agent Team Topology")
    st.markdown(
        """
        - 🧭 **Supervisor** (Router & Guardrail)
        - 🔍 **Researcher** (Search & Notes)
        - 📊 **Analyst** (Trade-offs & Insights)
        - ✍️ **Writer** (Synthesis & Report)
        - 🛡️ **Critic** (Fact-check & Audit)
        """
    )

# Main Application Tabs
st.markdown('<div class="main-header">Multi-Agent Research System</div>', unsafe_allow_html=True)
st.caption("Production-grade Multi-Agent Research & Benchmark Studio with LangGraph & Langfuse")

tab_research, tab_benchmark, tab_corpus, tab_architecture = st.tabs(
    [
        "🚀 Live Interactive Research",
        "📊 Benchmark & Comparison",
        "📚 Offline Corpus Explorer (30 Topics)",
        "🏗️ Workflow Architecture",
    ]
)

# TAB 1: LIVE INTERACTIVE RESEARCH & DUAL COMPARISON
with tab_research:
    st.subheader("⚔️ So Sánh Trực Quan 1-Click: Single-Agent vs Multi-Agent")
    st.markdown(
        "Nhập câu hỏi hoặc chọn từ **30 topics chuẩn**, nhấn nút bên dưới để hệ thống "
        "tự động kích hoạt song song **Single-Agent Baseline** và **Multi-Agent Workflow** "
        "và hiển thị bảng so sánh đối đầu trực quan."
    )

    col_input1, col_input2 = st.columns([3, 1])

    with col_input1:
        selected_topic = st.selectbox(
            "Chọn Topic từ bộ 30 Benchmark Corpus (hoặc nhập câu hỏi tùy ý bên dưới):",
            options=["-- Tùy chỉnh câu hỏi --"] + list(topics_map.keys()),
        )

        default_query = "Research GraphRAG state-of-the-art and write a comprehensive technical summary"
        if selected_topic != "-- Tùy chỉnh câu hỏi --":
            t_path = topics_map[selected_topic]
            with open(t_path, encoding="utf-8") as f:
                t_data = json.load(f)
            default_query = t_data.get("topic", {}).get("research_question", default_query)

        query_input = st.text_area(
            "Câu hỏi nghiên cứu (Research Query):",
            value=default_query,
            height=90,
            help="Nhập prompt nghiên cứu để gửi cho các Agent.",
        )

    with col_input2:
        run_mode = st.radio(
            "Chế độ thực thi:",
            [
                "⚔️ So sánh đối đầu (Cả 2)",
                "🤖 Chỉ chạy Multi-Agent",
                "👤 Chỉ chạy Single-Agent",
            ],
        )
        audience = st.selectbox(
            "Đối tượng độc giả:",
            ["technical learners", "AI engineers & researchers", "executive decision makers"],
        )

    btn_run = st.button("🚀 Chạy So Sánh Ngay", type="primary", use_container_width=True)

    if btn_run and query_input:
        if run_mode == "⚔️ So sánh đối đầu (Cả 2)":
            st.info("🔄 Đang chạy song song Single-Agent Baseline và Multi-Agent System...")

            with st.spinner("1/2: Đang thực thi Single-Agent Baseline..."):
                base_state, base_m = run_benchmark("Single-Agent", query_input, run_single_agent_baseline)

            with st.spinner("2/2: Đang thực thi Multi-Agent Workflow (Supervisor ➔ Researcher ➔ Analyst ➔ Writer)..."):
                multi_state, multi_m = run_benchmark("Multi-Agent", query_input, run_multi_agent_workflow)

            st.success("✅ Đã hoàn thành so sánh cả 2 hệ thống!")

            # 1. Bảng số liệu KPI so sánh trực quan
            st.markdown("### 📊 1. Bảng So Sánh Chỉ Số Hiệu Năng (KPIs)")
            k1, k2, k3, k4 = st.columns(4)

            # Latency
            lat_diff = multi_m.latency_seconds - base_m.latency_seconds
            k1.metric(
                label="⚡ Độ Trễ (Latency)",
                value=f"{multi_m.latency_seconds:.2f}s",
                delta=f"{lat_diff:+.2f}s (Overhead)",
                delta_color="inverse",
                help="Multi-Agent cần thời gian chuyển giao giữa các Agent.",
            )

            # Cost
            b_cost = base_m.estimated_cost_usd or 0.0
            m_cost = multi_m.estimated_cost_usd or 0.0
            cost_diff = m_cost - b_cost
            k2.metric(
                label="💰 Chi Phí Token (USD)",
                value=f"${m_cost:.4f}",
                delta=f"{cost_diff:+.4f} USD",
                delta_color="inverse",
                help="Multi-Agent tiêu tốn nhiều token hơn do phân tích nhiều bước.",
            )

            # Quality
            b_q = base_m.quality_score or 0.0
            m_q = multi_m.quality_score or 0.0
            q_diff = m_q - b_q
            k3.metric(
                label="🎯 Chất Lượng (Quality)",
                value=f"{m_q:.1f}/10",
                delta=f"{q_diff:+.1f} pts (Vượt trội)",
                help="Độ sâu kỹ thuật, phân tích trade-offs và cấu trúc bài viết.",
            )

            # Citation Coverage
            b_cit = (base_m.citation_coverage or 0.0) * 100
            m_cit = (multi_m.citation_coverage or 0.0) * 100
            cit_diff = m_cit - b_cit
            k4.metric(
                label="📚 Độ Phủ Nguồn (Citations)",
                value=f"{m_cit:.0f}%",
                delta=f"{cit_diff:+.0f}% (Xác thực)",
                help="Tỷ lệ trích dẫn nguồn chuẩn xác, hạn chế hallucination.",
            )

            # 2. Biểu đồ trực quan
            chart_df = pd.DataFrame(
                [
                    {
                        "Hệ thống": "Single-Agent Baseline",
                        "Độ trễ (giây)": round(base_m.latency_seconds, 2),
                        "Chi phí ($x1000)": round(b_cost * 1000, 3),
                        "Chất lượng (/10)": round(b_q, 1),
                        "Trích dẫn (%)": round(b_cit, 1),
                    },
                    {
                        "Hệ thống": "Multi-Agent System",
                        "Độ trễ (giây)": round(multi_m.latency_seconds, 2),
                        "Chi phí ($x1000)": round(m_cost * 1000, 3),
                        "Chất lượng (/10)": round(m_q, 1),
                        "Trích dẫn (%)": round(m_cit, 1),
                    },
                ]
            )

            c_col1, c_col2 = st.columns(2)
            with c_col1:
                st.markdown("**So sánh Độ trễ & Chất lượng:**")
                st.bar_chart(chart_df, x="Hệ thống", y=["Độ trễ (giây)", "Chất lượng (/10)"])
            with c_col2:
                st.markdown("**So sánh Độ phủ trích dẫn (%):**")
                st.bar_chart(chart_df, x="Hệ thống", y=["Trích dẫn (%)"])

            st.divider()

            # 3. So sánh Báo Cáo Song Song 2 Cột
            st.markdown("### 📄 2. So Sánh Báo Cáo Thực Tế (Side-by-Side Output)")
            col_left, col_right = st.columns(2)

            with col_left:
                st.markdown("#### 👤 Single-Agent Baseline Output")
                st.caption(f"Thời gian: {base_m.latency_seconds:.2f}s | Chi phí: ${b_cost:.4f}")
                st.info("⚠️ Trả lời trực tiếp từ 1 prompt: Thường bao quát, thiếu nguồn trích dẫn thật.")
                st.markdown(base_state.final_answer or "Không có câu trả lời.")

            with col_right:
                st.markdown("#### 🤖 Multi-Agent System Output")
                st.caption(f"Thời gian: {multi_m.latency_seconds:.2f}s | Chi phí: ${m_cost:.4f}")
                st.success("✅ Được tổng hợp qua Supervisor ➔ Researcher ➔ Analyst ➔ Writer:")
                st.markdown(multi_state.final_answer or "Không có câu trả lời.")

                with st.expander("🔍 Xem chi tiết Handoff & Ghi chú từng Agent"):
                    st.markdown("**Đường đi điều phối (Routing):**")
                    st.markdown(" ➔ ".join(f"`{r}`" for r in multi_state.route_history))

                    st.markdown("---")
                    st.markdown("**Ghi chú Researcher (Tài liệu nguồn):**")
                    st.markdown(multi_state.research_notes or "N/A")
                    if multi_state.sources:
                        for s in multi_state.sources:
                            st.markdown(f"- 🔗 [{s.title}]({s.url or '#'})")

                    st.markdown("---")
                    st.markdown("**Ghi chú Analyst (Phân tích Trade-offs & Failure Modes):**")
                    st.markdown(multi_state.analysis_notes or "N/A")

        elif run_mode == "🤖 Chỉ chạy Multi-Agent":
            with st.spinner("Đang chạy Multi-Agent Workflow..."):
                t0 = perf_counter()
                state = run_multi_agent_workflow(query_input, audience=audience)
                el = perf_counter() - t0

                st.success(f"Multi-Agent hoàn tất trong {el:.2f}s!")
                total_cost = sum(r.metadata.get("cost_usd", 0.0) or 0.0 for r in state.agent_results)

                st.metric("Độ trễ", f"{el:.2f}s")
                st.metric("Chi phí", f"${total_cost:.4f}")
                st.markdown("### 📄 Báo Cáo Hoàn Chỉnh")
                st.markdown(state.final_answer or "")

        else:  # Chỉ chạy Single-Agent
            with st.spinner("Đang chạy Single-Agent Baseline..."):
                t0 = perf_counter()
                state = run_single_agent_baseline(query_input, audience=audience)
                el = perf_counter() - t0

                st.success(f"Single-Agent hoàn tất trong {el:.2f}s!")
                cost = state.agent_results[0].metadata.get("cost_usd", 0.0) if state.agent_results else 0.0
                st.metric("Độ trễ", f"{el:.2f}s")
                st.metric("Chi phí", f"${cost:.4f}")
                st.markdown("### 📄 Báo Cáo Single-Agent")
                st.markdown(state.final_answer or "")

# TAB 2: BENCHMARK & COMPARISON
with tab_benchmark:
    st.subheader("Offline Corpus Benchmark Suite")
    st.markdown(
        "Evaluate and compare **Single-Agent Baseline vs Multi-Agent Workflow** across the standard benchmark topics."
    )

    bench_col1, bench_col2 = st.columns([2, 2])
    with bench_col1:
        num_topics = st.slider("Select number of topics to benchmark:", min_value=1, max_value=30, value=3)
    with bench_col2:
        out_filename = st.text_input("Output Report File:", value="reports/benchmark_report.md")

    if st.button("▶️ Execute Benchmark Suite", type="primary"):
        with st.spinner(f"Running automated benchmark on {num_topics} topics..."):
            metrics_list = run_corpus_benchmark(corpus_dir=CORPUS_DIR, limit=num_topics)
            report_text = render_markdown_report(metrics_list)

            # Save report
            out_p = Path(out_filename)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_text(report_text, encoding="utf-8")

            st.success(f"Benchmark completed! Report saved to `{out_filename}`.")

            # Prepare visualization DataFrame
            records = []
            for m in metrics_list:
                records.append(
                    {
                        "Run": m.run_name,
                        "Type": "Baseline" if "baseline" in m.run_name.lower() else "Multi-Agent",
                        "Latency (s)": round(m.latency_seconds, 2),
                        "Cost (USD)": round(m.estimated_cost_usd or 0.0, 4),
                        "Quality (0-10)": round(m.quality_score or 0.0, 1),
                        "Citation Cov (%)": round((m.citation_coverage or 0.0) * 100, 1),
                        "Notes": m.notes,
                    }
                )
            df = pd.DataFrame(records)

            st.markdown("### 📊 Metrics Table")
            st.dataframe(df, use_container_width=True)

            # Charts
            st.markdown("### 📈 Visual Comparisons")
            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                st.markdown("**Latency & Quality Comparison**")
                st.bar_chart(df, x="Run", y=["Latency (s)", "Quality (0-10)"])
            with chart_col2:
                st.markdown("**Cost & Citation Coverage**")
                st.bar_chart(df, x="Run", y=["Citation Cov (%)"])

            st.divider()
            st.markdown("### 📝 Full Markdown Report")
            st.markdown(report_text)
            st.download_button(
                "📥 Download Benchmark Report (.md)",
                data=report_text,
                file_name="benchmark_report.md",
                mime="text/markdown",
            )

# TAB 3: CORPUS EXPLORER
with tab_corpus:
    st.subheader("Offline Knowledge Corpus Explorer (30 Topics)")
    st.caption("Inspect embedded knowledge articles, fact bank, and 100-point rubric for each topic.")

    sel_explore = st.selectbox("Choose a topic to inspect:", options=list(topics_map.keys()))

    if sel_explore:
        t_file = topics_map[sel_explore]
        with open(t_file, encoding="utf-8") as f:
            corpus_data = json.load(f)

        meta = corpus_data.get("benchmark_metadata", {})
        topic_info = corpus_data.get("topic", {})
        kb = corpus_data.get("knowledge_base", {})
        task = corpus_data.get("research_task", {})

        c1, c2, c3 = st.columns(3)
        c1.metric("Topic ID", meta.get("topic_id", ""))
        c2.metric("Difficulty", meta.get("difficulty", "advanced"))
        c3.metric("Target Words", f"{topic_info.get('expected_report_length_words', {}).get('target', 2800)}w")

        st.markdown(f"**Research Question:** {topic_info.get('research_question', '')}")
        st.markdown(f"**Working Thesis:** {topic_info.get('working_thesis_for_evaluation', '')}")

        st.divider()

        # Knowledge Articles
        with st.expander(f"📚 Knowledge Articles ({len(kb.get('knowledge_articles', []))} articles)", expanded=True):
            for art in kb.get("knowledge_articles", []):
                st.markdown(f"#### [{art.get('article_id')}] {art.get('title')}")
                st.markdown(art.get("content", ""))
                st.divider()

        # Fact Bank
        with st.expander(f"🔑 Fact Bank ({len(kb.get('fact_bank', []))} atomic facts)"):
            facts = kb.get("fact_bank", [])
            for fact in facts:
                st.markdown(f"- **[{fact.get('fact_id')}]**: {fact.get('statement')}")

        # Evaluation Rubric (100 pts)
        with st.expander("📋 100-Point Evaluation Rubric"):
            rubric = task.get("evaluation_rubric", [])
            r_df = pd.DataFrame(rubric)
            st.dataframe(r_df, use_container_width=True)

# TAB 4: ARCHITECTURE
with tab_architecture:
    st.subheader("Multi-Agent Architecture & Orchestration")

    st.markdown(
        """
        ```mermaid
        graph TD
            Start([START]) --> Sup[🧭 Supervisor Router]
            Sup -->|Missing Research| Res[🔍 Researcher Agent]
            Sup -->|Missing Analysis| Ana[📊 Analyst Agent]
            Sup -->|Missing Final Answer| Wri[✍️ Writer Agent]
            Sup -->|Optional Review| Cri[🛡️ Critic Agent]
            Sup -->|Answer Complete or Max Iterations| Done([END / Complete])

            Res -->|Update research_notes & sources| Sup
            Ana -->|Update analysis_notes| Sup
            Wri -->|Update final_answer| Sup
            Cri -->|Audit findings| Sup
        ```
        """
    )

    st.markdown(
        """
        ### Core Design Principles
        1. **Single Source of Truth**: `ResearchState` (Pydantic model) is passed across all agents without context loss.
        2. **Stateful Handoffs & Supervisor Routing**: Supervisor inspects state at every turn and directs control conditionally.
        3. **Guardrails**: Hard limit on `max_iterations` to prevent infinite loops, combined with LLM cost/token tracking.
        4. **Observability**: Real-time tracing sent to Langfuse Cloud for performance analysis and debugging.
        """
    )
