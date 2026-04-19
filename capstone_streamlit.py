
# capstone_streamlit.py
# Launch: streamlit run capstone_streamlit.py
import uuid, streamlit as st
from agent import app, ask, DOCUMENTS

st.set_page_config(page_title="Agentic AI Course Assistant", page_icon="🤖", layout="wide")

@st.cache_resource
def get_agent():
    return app

agent_app = get_agent()

if "messages"   not in st.session_state: st.session_state.messages   = []
if "thread_id"  not in st.session_state: st.session_state.thread_id  = str(uuid.uuid4())

with st.sidebar:
    st.title("🤖 Course Assistant")
    st.subheader("📚 Topics Covered")
    for d in DOCUMENTS: st.markdown(f"- {d['topic']}")  
    st.markdown("---")
    if st.button("🔄 New Conversation", use_container_width=True):
        st.session_state.messages  = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

st.title("🎓 Agentic AI Course Assistant")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Ask a course question…"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"): st.markdown(user_input)
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            result = ask(user_input, thread_id=st.session_state.thread_id)
        st.markdown(result["answer"])
        cols = st.columns(3)
        cols[0].caption(f"🔀 Route: `{result['route']}`")
        cols[1].caption(f"🎯 Faithfulness: `{result['faithfulness']:.2f}`")
    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
