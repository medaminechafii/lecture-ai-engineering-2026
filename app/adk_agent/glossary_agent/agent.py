import os

from google.adk.agents import Agent
from app import tools
from app.config import MODEL

if "GEMINI_API_KEY" in os.environ:
    os.environ.setdefault("GOOGLE_API_KEY", os.environ["GEMINI_API_KEY"])

INSTRUCTIONS = """\
あなたはAIエンジニアリング講座（AIE2026）の学習支援エージェントです。
- 用語や講座に固有のことを答える前に、必ず search_glossary で用語集を確認する。
- 回の内容を聞かれたら get_lecture_summary を使う。
- 確認問題やクイズを求められたら make_quiz を使う。
- 用語集に見つからないことは、推測で断定せず「用語集にはない」と伝える。
- 回答の最後に、関係する回（第N回）を示す。
- 宿題の答えそのものは書かず、考え方のヒントで導く。
"""

root_agent = Agent(
    model=MODEL,
    name="GlossaryAgent",
    instruction=INSTRUCTIONS,
    tools=[
        tools.search_glossary,
        tools.get_lecture_summary,
        tools.make_quiz,
    ],
)