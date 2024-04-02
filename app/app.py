import gradio as gr
import whisper
from newspaper import Article
from transformers import AutoTokenizer, BartForConditionalGeneration
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI'))

model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
whisper_model = whisper.load_model('base.en')

# News links (hardcoded) it works also with new links, but we didn't add to the MVP
news_data = {
    1: {"title": "Novak Djokovic vs Jannik Sinner",
        "url": "https://edition.cnn.com/2024/01/26/sport/jannik-sinner-novak-djokovic-australian-open-spt-intl/index.html"},
    2: {"title": "Apple’s Vision Pro headset hits US stores today",
        "url": "https://edition.cnn.com/2024/02/02/tech/apple-vision-pro-what-you-need-to-know/index.html"},
    3: {"title": "Russia's war capacity endures for two-three years.",
        "url": "https://edition.cnn.com/2024/02/14/europe/russia-sustain-war-effort-ukraine-analysis-intl/index.html"}
}

summary_dicc = {
    'Summary 1': '',
    'Summary 2': '',
    'Summary 3': ''
}

final_summary = ''

final_script = ''

transcripts = {
    1: ["", ""],
    2: ["", ""],
    3: ["", ""]
}

scripts = {}

css = """
h1 {
    text-align: center;
    display:block;
}
h2 {
    text-align: center;
    display:block;
}
h3 {
    text-align: center;
    display:block;
}
"""


# Get the link from news_data using the value of the button from gradio
def get_news_url(news_index):
    index = int(news_index.split(' ')[1])
    return news_data[index]['url']


# Extract the article using Newspaper 3k
def create_article(url):
    article = Article(url)
    article.download()
    article.parse()
    return article


# Generates 3 summaries using BERT, we change the temperature and other parameters to change the style of each summary
def summary_generator(selected_news):
    global summary_dicc

    article = create_article(selected_news)
    ARTICLE_TO_SUMMARIZE = (
        article.text
    )
    inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors='pt', truncation=True)
    summaries_ = []
    for i in range(3):
        summary_ids = model.generate(inputs["input_ids"],
                                     num_beams=4 + i,  # fixed number of beams
                                     temperature=1.0 + (0.3 * i),  # increasing temperature
                                     top_k=50 + (20 * i),  # increasing top_k
                                     min_length=100,
                                     max_length=200,
                                     do_sample=True,
                                     early_stopping=True)

        summary = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        summaries_.append(summary)

    for i, key in enumerate(summary_dicc.keys()):
        summary_dicc[key] = summaries_[i]

    return summaries_[0], summaries_[1], summaries_[2]


# Selects the summary from the global variable
def summary_selection(selected_summary):
    global final_summary
    final_summary = summary_dicc[selected_summary]
    return final_summary


# Selects the script from the global variable
def script_selection(selected_script):
    global final_script
    index = int(selected_script.split(' ')[-1])
    final_script = scripts[index]
    return final_script


# Counts the number of words of the transcript to use it in the prompt
def word_counter(sample_text):
    text_length = len(sample_text.split())
    return text_length


# Generates the transcription from the video using whisper and add it to the global variable
def transcribe(number, file_path):
    transcription = whisper_model.transcribe(file_path, fp16=True)
    key = int(number.split(' ')[-1])
    transcripts[key][0] = transcription['text']
    transcripts[key][1] = word_counter(transcripts[key][0])
    return transcription['text']


# Final prompt to create the personalized scripts
def generate_prompt_final():
    # Including the news summary in the prompt
    prompt_text = f"""
    Objective
    To analyze provided example transcripts for speaking patterns and stylistic features, generate new text on a given topic as if spoken by the person from the transcripts, and evaluate the generated text against the learned patterns and metrics, ensuring alignment with dynamic speech timing.
    Step 1: Preparation
    • Receive example transcripts of a person speaking on various topics.
    • Given topics for new text generation.
    Step 2: Analysis of Transcripts
    • Identify and understand the unique speaking patterns, style, tone, and features of the speaker.
    • Catalogue speaking patterns including specific phrases, vocabulary, idiomatic expressions, sentence structure, and tone. Here are the trancripts:
    {transcripts}
    Step 3: Calculation of Dynamic Target Word Count
    • Calculate the Words Per Minute (WPM) for each transcript.
    • Determine the average WPM across all provided examples.
    • Based on the average WPM, calculate the target word count for the new text, aiming for a specific speech length (e.g., 22 seconds), making the "Clear Target Word Count" dynamic and adaptable.
    • Ensure you do the math step by step to ensure accuracy.
    Step 4: Metric-Based Learning
    • Use detailed metrics (Lexical Diversity, Sentence Length, Tone Analysis, etc.) to quantitatively and qualitatively analyze the transcripts.
    • Provide scores for each metric to quantify the speaking style.
    Step 5: Text Generation
    •	Generate three unique and dynamic new texts on the provided topic, ensuring each matches the identified speaking patterns, style, and metrics and fits within the dynamically calculated target word count for the specified speech length. Use clear markers to delineate each generated text for easy extraction. 
    •	Please mark each generated text with the following markers for easy extraction:
    •	For the first text, use "---Generated Text 1 Start---" and "---Generated Text 1 End---".
    •	For the second text, use "---Generated Text 2 Start---" and "---Generated Text 2 End---".
    •	For the third text, use "---Generated Text 3 Start---" and "---Generated Text 3 End---".


    News Summary:
    {final_summary}

    Please mark each generated text with clear separators for easy identification and extraction.
    """
    return prompt_text


# Calls the GPT API to generate the scripts using the prompt above
def callAPI():
    prompt = generate_prompt_final()
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
    )
    result = extract_generated_texts(response.choices[0].message.content)
    return result[1], result[2], result[3]


# Extract each script from the final prompt
def extract_generated_texts(input_text):
    markers = [
        ("---Generated Text 1 Start---", "---Generated Text 1 End---"),
        ("---Generated Text 2 Start---", "---Generated Text 2 End---"),
        ("---Generated Text 3 Start---", "---Generated Text 3 End---")
    ]

    for i, (start_marker, end_marker) in enumerate(markers, start=1):
        start_index = input_text.find(start_marker) + len(start_marker)
        end_index = input_text.find(end_marker)
        if start_index > len(start_marker) and end_index > -1:
            extracted_text = input_text[start_index:end_index].strip()
            scripts[i] = extracted_text

    return scripts


with (gr.Blocks(css=css) as demo):
    gr.Markdown('# WELCOME TO TRENDIFY')
    with gr.Row():
        with gr.Column(scale=2):
            title1 = news_data[1]['title']
            gr.Markdown(f'### {title1}')
            gr.Image(value="pictures/sinner.jpg", label=news_data[1]['title'])
        with gr.Column(scale=2):
            title2 = news_data[2]['title']
            gr.Markdown(f'### {title2}')
            gr.Image(value="pictures/apple_vision.jpg", label=news_data[2]['title'])
        with gr.Column(scale=2):
            title3 = news_data[3]['title']
            gr.Markdown(f'### {title3}')
            gr.Image(value="pictures/war.jpg", label=news_data[3]['title'])

    news = ['News 1', 'News 2', 'News 3']
    news_radio = gr.Radio(news, label='Trendy News')
    news_btn = gr.Button('Select News')
    output_news = gr.Textbox(label='News Selected')
    news_btn.click(fn=get_news_url, inputs=news_radio, outputs=output_news)
    generate_summary_btn = gr.Button("Generate Summary")
    # radio_summary = gr.Radio(summary_options, label='Select Summary', interactive=True)
    s1 = gr.Textbox(label='Summary 1', show_copy_button=True)
    s2 = gr.Textbox(label='Summary 2', show_copy_button=True)
    s3 = gr.Textbox(label='Summary 3', show_copy_button=True)
    generate_summary_btn.click(fn=summary_generator, inputs=output_news, outputs=[s1, s2, s3])

    # Select summary
    summaries = ['Summary 1', 'Summary 2', 'Summary 3']
    summary_radio = gr.Radio(summaries, label='Select a summary')
    select_summary_btn = gr.Button('Select your Summary')
    output_summary = gr.Textbox(label='Selected Summary', show_copy_button=True)
    select_summary_btn.click(fn=summary_selection, inputs=summary_radio, outputs=output_summary)

    # Upload video
    # Second Section - Upload Video and Transcript
    with gr.Group():
        gr.Markdown("# Upload your Videos")
        gr.Markdown("### Upload 3 videos, so we can extract your style and help you generate new content!")

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown('## Video 1')
            video_upload_1 = gr.Video(label="Upload First Video")
            video_transcript_1 = gr.Textbox(label='Video 1 Transcript')
            video_upload_btn_1 = gr.Button("Upload Video 1", value=1)
            video_upload_btn_1.click(fn=transcribe, inputs=[video_upload_btn_1, video_upload_1],
                                     outputs=video_transcript_1)
        with gr.Column(scale=2):
            gr.Markdown('## Video 2')
            video_upload_2 = gr.Video(label="Upload Second Video")
            video_transcript_2 = gr.Textbox(label='Video 2 Transcript')
            video_upload_btn_2 = gr.Button("Upload Video 2")
            video_upload_btn_2.click(fn=transcribe, inputs=[video_upload_btn_2, video_upload_2],
                                     outputs=video_transcript_2)
        with gr.Column(scale=2):
            gr.Markdown('## Video 3')
            video_upload_3 = gr.Video(label="Upload Third Video")
            video_transcript_3 = gr.Textbox(label='Video 3 Transcript')
            video_upload_btn_3 = gr.Button("Upload Video 3")
            video_upload_btn_3.click(fn=transcribe, inputs=[video_upload_btn_3, video_upload_3],
                                     outputs=video_transcript_3)

    # Third Section - Generation and Selection of Script
    with gr.Group():
        gr.Markdown("# Script Generation")
        gr.Markdown("### Now we will generate some options of a script for your new video. Choose your favorite!")
    generate_script = gr.Button('Generate Scripts')
    sc1 = gr.Textbox(label='Script 1')
    sc2 = gr.Textbox(label='Script 2')
    sc3 = gr.Textbox(label='Script 3')
    generate_script.click(fn=callAPI, outputs=[sc1, sc2, sc3])

    scripts_list = ['Script 1', 'Script 2', 'Script 3']
    script_radio = gr.Radio(scripts_list, label='Select a script')
    select_script_btn = gr.Button('Select your Script')
    output_script = gr.Textbox(label='Selected Script', show_copy_button=True)
    select_script_btn.click(fn=script_selection, inputs=script_radio, outputs=output_script)

if __name__ == "__main__":
    demo.launch()
