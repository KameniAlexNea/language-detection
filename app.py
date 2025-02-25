import gradio as gr
import torch
from transformers import pipeline
import pycountry

# Load model and tokenizer
model_name = "alexneakameni/language_detection"
device = "cuda" if torch.cuda.is_available() else "cpu"

language_detection_pipeline = pipeline(
    "text-classification", model=model_name, device=0 if device == "cuda" else -1
)

sentences = [
    # English
    "Although artificial intelligence has made significant progress in recent years, there are still many challenges to overcome before it can truly replicate human intelligence.",
    # French
    "Bien que l'intelligence artificielle ait fait des progrès considérables ces dernières années, de nombreux défis restent à relever avant qu'elle ne puisse véritablement imiter l'intelligence humaine.",
    # Spanish
    "A pesar de los importantes avances en inteligencia artificial en los últimos años, aún existen muchos desafíos por superar antes de que pueda replicar verdaderamente la inteligencia humana.",
    # German
    "Obwohl künstliche Intelligenz in den letzten Jahren erhebliche Fortschritte gemacht hat, gibt es noch viele Herausforderungen, die überwunden werden müssen, bevor sie die menschliche Intelligenz wirklich nachbilden kann.",
    # Italian
    "Sebbene l'intelligenza artificiale abbia fatto progressi significativi negli ultimi anni, ci sono ancora molte sfide da affrontare prima che possa davvero replicare l'intelligenza umana.",
    # Portuguese
    "Embora a inteligência artificial tenha avançado significativamente nos últimos anos, ainda há muitos desafios a superar antes que ela possa realmente imitar a inteligência humana.",
    # Dutch
    "Hoewel kunstmatige intelligentie de afgelopen jaren aanzienlijke vooruitgang heeft geboekt, zijn er nog veel uitdagingen te overwinnen voordat het echt menselijke intelligentie kan nabootsen.",
    # Russian
    "Несмотря на значительный прогресс в области искусственного интеллекта в последние годы, все еще остается много проблем, которые необходимо решить, прежде чем он сможет действительно имитировать человеческий интеллект.",
    # Arabic
    "على الرغم من التقدم الكبير الذي أحرزته الذكاء الاصطناعي في السنوات الأخيرة، لا تزال هناك العديد من التحديات التي يجب التغلب عليها قبل أن يتمكن من محاكاة الذكاء البشري حقًا.",
    # Hindi
    "हालांकि कृत्रिम बुद्धिमत्ता ने हाल के वर्षों में उल्लेखनीय प्रगति की है, फिर भी कई चुनौतियाँ बनी हुई हैं जिन्हें पार किए बिना यह वास्तव में मानव बुद्धिमत्ता की नकल नहीं कर सकती।",
    # Chinese (Simplified)
    "尽管近年来人工智能取得了重大进展，但仍然存在许多挑战，需要克服这些挑战才能真正复制人类智能。",
    # Japanese
    "近年、人工知能は大きな進歩を遂げましたが、人間の知能を本当に再現するにはまだ多くの課題を克服する必要があります。",
    # Korean
    "인공지능이 최근 몇 년 동안 상당한 발전을 이루었음에도 불구하고, 인간의 지능을 진정으로 재현하기 위해서는 아직 극복해야 할 많은 도전 과제가 남아 있습니다.",
    # Turkish
    "Yapay zeka son yıllarda önemli ilerlemeler kaydetmiş olsa da, insan zekasını gerçekten taklit edebilmesi için hala birçok zorluk aşılmalıdır.",
    # Polish
    "Chociaż sztuczna inteligencja poczyniła w ostatnich latach znaczne postępy, nadal istnieje wiele wyzwań do pokonania, zanim będzie mogła naprawdę naśladować ludzką inteligencję.",
    # Greek
    "Αν και η τεχνητή νοημοσύνη έχει σημειώσει σημαντική πρόοδο τα τελευταία χρόνια, εξακολουθούν να υπάρχουν πολλές προκλήσεις που πρέπει να ξεπεραστούν πριν μπορέσει πραγματικά να αναπαραγάγει την ανθρώπινη νοημοσύνη.",
    # Hebrew
    "למרות שהבינה המלאכותית התקדמה באופן משמעותי בשנים האחרונות, עדיין ישנם אתגרים רבים שיש להתגבר עליהם לפני שתוכל באמת לשחזר את האינטליגנציה האנושית.",
    # Swahili
    "Ingawa akili bandia imepiga hatua kubwa katika miaka ya hivi karibuni, bado kuna changamoto nyingi zinazopaswa kushindwa kabla ya kuweza kuiga akili ya binadamu kwa kweli.",
    # Vietnamese
    "Mặc dù trí tuệ nhân tạo đã đạt được những tiến bộ đáng kể trong những năm gần đây, nhưng vẫn còn nhiều thách thức cần vượt qua trước khi nó có thể thực sự tái tạo trí thông minh của con người.",
    # Thai
    "แม้ว่าปัญญาประดิษฐ์จะมีความก้าวหน้าอย่างมากในช่วงไม่กี่ปีที่ผ่านมา แต่ยังคงมีความท้าทายอีกมากที่ต้องเอาชนะก่อนที่มันจะสามารถเลียนแบบสติปัญญาของมนุษย์ได้อย่างแท้จริง.",
]


def get_language_name(code: str):
    lang = code.split("_")[0]  # Extract the first part before '_'
    try:
        return pycountry.languages.get(alpha_3=lang).name  # Get ISO 639-1
    except AttributeError:
        return lang  # Fallback to original if no match


def predict_language(text, top_k=5):
    """Predicts the top-k languages for the given text."""
    results = language_detection_pipeline(text, top_k=top_k)
    formatted_results = [
        f"{get_language_name(result['label'])} - {result['label']}: {result['score']:.4f}"
        for result in results
    ]
    return "\n".join(formatted_results)


# Create Gradio interface
demo = gr.Interface(
    fn=predict_language,
    inputs=[
        gr.Textbox(label="Enter text", placeholder="Type a sentence here..."),
        gr.Slider(1, 10, value=5, step=1, label="Top-k Languages"),
    ],
    outputs=gr.Textbox(label="Predicted Languages"),
    title="🌍 Language Detection",
    description="Detects the language of a given text using a fine-tuned BERT model. Returns the top-k most probable languages.",
    examples=[[sent, 5] for sent in sentences],
    flagging_mode="manual"
)

demo.launch()