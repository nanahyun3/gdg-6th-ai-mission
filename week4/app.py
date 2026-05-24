
import gradio as gr
from transformers import pipeline

classifier = pipeline("text-classification", model="nanahyun3/nsmc-sentiment")

def format_result(result):
    label = result["label"]
    score = result["score"]

    if label == "LABEL_1":
        emoji, label_kr = "😊", "긍정"
    else:
        emoji, label_kr = "😞", "부정"

    return f"{emoji} {label_kr} (확신도: {score:.1%})"

def predict(text):
    if not text.strip():
        return "문장을 입력해주세요."
    result = classifier(text)[0]
    return format_result(result)

demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(label="영화 리뷰", placeholder="리뷰를 입력하세요...", lines=3),
    outputs=gr.Textbox(label="감정 분석 결과"),
    title="AI 영화 리뷰 감정 분석기",
    description="NSMC 데이터로 파인튜닝된 한국어 감정 분석 모델입니다.",
    examples=[
        ["이 영화 진짜 재미있어요!"],
        ["완전 지루하고 별로였음"],
        ["배우 연기는 좋았지만 스토리가 아쉬웠다"]
    ]
)
demo.launch()
