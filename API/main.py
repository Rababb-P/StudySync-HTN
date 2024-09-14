import boto3
from botocore.exceptions import NoCredentialsError
import cohere
import re
from flask import Flask, request, jsonify
from openai import OpenAI
import requests
import io

from flask_cors import CORS


QUESTION_PATTERN = r"^(\d+\.\s(?![A-D]\.).+?[.?,:]$)"  # Match the question
OPTION_PATTERN = r"^([A-D]\.\s.+)$"  # Match each option
ANSWER_MATCH = r"Answer:\s*([A-D])" #Match the answer

app = Flask(__name__)
CORS(app)
#/Users/manveersohal/Documents/GitHub/group-study-app-htn/.venv/bin/python -m pip install flask 
quiz_prompt="""Generated Quiz:



1. What is the base of ice cream?
A. Milk
B. Water
C. Juice
D. Nitrogen

2. What is the most common flavor of ice cream?
A. Strawberry
B. Vanilla
C. Chocolate
D. Ginger

3. What is the purpose of adding sweeteners, spices, fruits, and stabilizers to the ice cream base?
A. To make the ice cream healthier
B. To make the ice cream taste and feel better
C. To make the ice cream sweeter
D. To make the ice cream last longer

4. Who is the ideal ice cream consumer?
A. People with dairy allergies or vegans
B. People with gluten allergies or vegetarians
C. People with sweet teeth
D. People with sour teeth

5. Which of the following milks can be used in ice cream?
A. Cow's milk
B. Almond milk
C. Oat milk
D. All of the above

### Answers

1. A: Milk
2. B: Vanilla
3. B: To make the ice cream taste and feel better
4. A: People with dairy allergies or vegans
5. D: All of the above 

 hope these answers are what you're looking for! Let me know if you would like me to explain any of the answers in detail. """


"""
How this basically will look like once created and made

self.question:
1. Ice cream is characterized by a ________ temperature range, from very low storage temperatures to increased temperatures for serving.

self.options:
[['A. Narrow'], ['B. Moderate'], ['C. Wide'], ['D. Variable']]

self.answer:
C

I created this object so when sending it to the front end, it is easier the use the data

"""
class Question:
    def __init__(self,question,):
        self.question = question
        self.options = []
        self.answer = None


    def __str__(self):
        return f'({self.question}, {self.options},{self.answer})'

    def add_option(self,option):
        self.options.append(option)

    def add_answer(self, answer):
        self.answer = answer

    def to_dict(self): 
        # Convert the object to a dictionary
        return  {
             'question': self.question, 
             'options': self.options, 
             'answer': self.answer 
             }












co = cohere.Client("")


"""
Gets a string of text and converts it into bullet points of key ideas

Uses cohere's API to summarize the text
"""

def text_to_bullet_list(text):
    print(len(text))
    if(len(text) < 250):
        return 
    
    response = co.summarize(
        text=text,
        format="bullets",
    )
    return response.summary



"""
Gets a string of text (in bullet form) and converts it into a quiz with questions and answers

Uses cohere's API to create the quiz
"""
def bullet_list_to_quiz(text):
    prompt = f"""
    Based on the following content, create a multiple-choice quiz. 
    - Each question should be numbered (1., 2., etc.).
    - Each question should have up to 4 answer options, labeled A., B., C., and D.
    - Provide the correct answer after each question. 
    - The answer should just be the correct letter
    - Do not include any extra text other than the questions, options, and answers.
    - The format should be strictly followed
    - always end the questions with a punctuation such as , . ? :
    - try to make around 5 questions
    - The question and the question number has to be on the same line
    - The format should be:

    1. Question text
    A. Option 1
    B. Option 2
    C. Option 3
    D. Option 4
    Answer: (this would be the letter).

    Content: {text}

    Quiz:b  
    """

    # Use Cohere to generate the quiz
    response = co.generate(
        prompt=prompt,
        max_tokens=400
    )

    # Print the generated quiz
    print("Generated Quiz:\n")
    print(response.generations[0].text)
    return response.generations[0].text


"""
Gets the prompted text and turns it into a usable format, where each question becomes its own object.
"""
def format_quiz(quiz_prompt):
    quiz = {}
    question_count = 0
    lines = quiz_prompt.split('\n')
    print(quiz_prompt)

    for line in lines:
        

        #identifies if the line is a question and stores it
        line = line.strip()
        question_match = re.search(QUESTION_PATTERN, line)
        question = question_match.group(1).strip() if question_match else None

        #if the line was a question, update the question count, and create a new Question object
        if (question):
            print("question: ",question)
            question_count += 1
            quiz[f'Question_{question_count}'] = Question(question)

        #identifies if the line is an option and stores it
        options_matches = re.findall(OPTION_PATTERN, line)
        options = [option.strip() for option in options_matches]

        #if the line was an option, add it to the question object's options
        if (options):
            print("options: ",options)
            quiz[f'Question_{question_count}'].add_option(options)

        answers_match = re.search(ANSWER_MATCH, line)
        answer = answers_match.group(1).strip() if answers_match else None
        if (answer):
            print("answer: ",answer)
            quiz[f'Question_{question_count}'].add_answer(answer[0])

    return quiz


#sample text
text = (
    "Ice cream is a sweetened frozen food typically eaten as a snack or dessert. "
    "It may be made from milk or cream and is flavoured with a sweetener, "
    "either sugar or an alternative, and a spice, such as cocoa or vanilla, "
    "or with fruit such as strawberries or peaches. "
    "It can also be made by whisking a flavored cream base and liquid nitrogen together. "
    "Food coloring is sometimes added, in addition to stabilizers. "
    "The mixture is cooled below the freezing point of water and stirred to incorporate air spaces "
    "and to prevent detectable ice crystals from forming. The result is a smooth, "
    "semi-solid foam that is solid at very low temperatures (below 2 °C or 35 °F). "
    "It becomes more malleable as its temperature increases.\n\n"
    'The meaning of the name "ice cream" varies from one country to another. '
    'In some countries, such as the United States, "ice cream" applies only to a specific variety, '
    "and most governments regulate the commercial use of the various terms according to the "
    "relative quantities of the main ingredients, notably the amount of cream. "
    "Products that do not meet the criteria to be called ice cream are sometimes labelled "
    '"frozen dairy dessert" instead. In other countries, such as Italy and Argentina, '
    "one word is used fo\r all variants. Analogues made from dairy alternatives, "
    "such as goat's or sheep's milk, or milk substitutes "
    "(e.g., soy, cashew, coconut, almond milk or tofu), are available for those who are "
    "lactose intolerant, allergic to dairy protein or vegan."
)

def for_fun(final):
    string= ""
    for key, question in final.items():
        print(question.question)
        print(question.options)
        print("answer", question.answer)
        string =string + f"<p>{question.question}</p> <p>{question.options}</p> <p>answer, {question.answer} </p>"

    return string


def start_transcription(temp_file_path):
    client = OpenAI(api_key="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")

    #audio_file= open("/Users/manveersohal/Documents/GitHub/group-study-app-htn/API/The 10 Second Rule #shorts.mp3", "rb")
    with open(temp_file_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
        model="whisper-1", 
        file=audio_file
        )
    os.remove(temp_file_path)
    print(transcription.text)
    return transcription.text

########################
########################
########################


import os
import io
import cv2
import time
import tempfile
import requests
from typing import Optional

API_URL_DEFAULT = "https://symphoniclabs--symphonet-vsr-modal-htn-model-upload-static-htn.modal.run"
API_URL = os.getenv("SYMPHONIC_API_URL", API_URL_DEFAULT)


def transcribe_bytes(video_bytes: bytes, *, timeout: int = 120) -> str:
    """
    Send raw MP4 bytes to the transcription API and return the raw response text.
    """
    if not video_bytes:
        raise ValueError("video_bytes is empty")

    files = {"video": ("input.mp4", io.BytesIO(video_bytes), "video/mp4")}
    try:
        resp = requests.post(API_URL, files=files, timeout=timeout)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as e:
        # Keep it commit-safe (no secrets), but still informative.
        return f"[error] request to API failed: {e}"


def transcribe_file(file_path: str, *, timeout: int = 120) -> str:
    """
    Read an MP4 file from disk and send it to the transcription API.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"No such file: {file_path}")

    with open(file_path, "rb") as f:
        return transcribe_bytes(f.read(), timeout=timeout)


def capture_and_transcribe_live(
    camera_index: int = 0,
    chunk_duration_sec: float = 4.0,
    frame_rate: int = 60,
    show_window: bool = True,
    stop_key: str = "q",
) -> None:
    """
    Capture webcam video in fixed-size chunks and send each chunk to the API.

    Press the `stop_key` while the display window is focused to exit.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("[error] Could not open webcam.")
        return

    # Probe a first frame to size the writer correctly
    ret, frame = cap.read()
    if not ret:
        print("[error] Could not read initial frame.")
        cap.release()
        return

    h, w = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    total_frames = max(1, int(chunk_duration_sec * frame_rate))

    print(f"[info] Using API_URL={API_URL}")
    print(f"[info] Capturing {total_frames} frames per chunk at {frame_rate} FPS ({chunk_duration_sec}s).")
    print(f"[info] Press '{stop_key}' to stop.")

    try:
        while True:
            # Optional live preview
            if show_window:
                cv2.imshow("Webcam Feed", frame)
                # Non-blocking check for stop key
                if cv2.waitKey(1) & 0xFF == ord(stop_key):
                    break

            # Record a chunk
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(prefix="chunk_", suffix=".mp4", delete=False) as tmp:
                    tmp_path = tmp.name

                out = cv2.VideoWriter(tmp_path, fourcc, frame_rate, (w, h))
                frames_written = 0

                # We already have one frame in `frame`; include it for snappier chunk start
                out.write(frame)
                frames_written += 1

                while frames_written < total_frames:
                    ret, frame = cap.read()
                    if not ret:
                        print("[warn] Could not read frame; ending capture loop.")
                        break
                    out.write(frame)
                    frames_written += 1

                out.release()

                # Send the chunk
                with open(tmp_path, "rb") as f:
                    chunk_bytes = f.read()

                started = time.time()
                transcription = transcribe_bytes(chunk_bytes)
                elapsed = time.time() - started

                print("\n======= TRANSCRIPTION RESULT =======")
                print(transcription.strip() or "[empty response]")
                print(f"============= ({elapsed:.2f}s) =============\n")

            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass

            # Grab next frame to keep the preview fresh
            ret, frame = cap.read()
            if not ret:
                print("[warn] Could not read frame after chunk; stopping.")
                break

    except KeyboardInterrupt:
        print("\n[info] Interrupted by user.")
    finally:
        cap.release()
        if show_window:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Chunked webcam transcription to Symphonic API.")
    sub = parser.add_subparsers(dest="cmd", required=False)

    # File mode
    p_file = sub.add_parser("file", help="Transcribe an existing MP4 file.")
    p_file.add_argument("path", help="Path to an MP4 file.")

    # Live mode
    p_live = sub.add_parser("live", help="Capture webcam and transcribe in chunks.")
    p_live.add_argument("--camera", type=int, default=0, help="Camera index (default 0).")
    p_live.add_argument("--chunk", type=float, default=4.0, help="Chunk duration in seconds (default 4).")
    p_live.add_argument("--fps", type=int, default=60, help="Frames per second (default 60).")
    p_live.add_argument("--no-window", action="store_true", help="Disable preview window.")
    p_live.add_argument("--stop-key", default="q", help="Key to stop (default 'q').")

    args = parser.parse_args()

    if args.cmd == "file":
        print(transcribe_file(args.path))
    else:
        # Default to live mode if no subcommand provided
        capture_and_transcribe_live(
            camera_index=getattr(args, "camera", 0),
            chunk_duration_sec=getattr(args, "chunk", 4.0),
            frame_rate=getattr(args, "fps", 60),
            show_window=not getattr(args, "no_window", False),
            stop_key=getattr(args, "stop_key", "q"),
        )




# Run the live transcription
#capture_and_transcribe_live()

########################

# def createBulletList(bullet_points):
#     bulletList = []
#     lines = bullet_points.split('\n')
#     print(lines)
#     return lines


@app.route('/upload', methods=['POST'])
def upload():
    if'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
   
    file = request.files['file']


    # If the user does not select a file
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    
    print("got the file!", request.files['file'])

    temp_file_path = os.path.join("", file.filename)
    file.save(temp_file_path)

    # Save the file to a directory (if needed)
    #file_path = os.path.join('uploads', file.filename)
    #file.save(file_path)
    print(file.filename)
    
    #audio file translated to transcript 
    response = start_transcription(temp_file_path)
    bullet_points = text_to_bullet_list(response)

    # bullet_points = '{<br/>}'.join(bullet_points.split('\n'))

    #turns into a list of points
    #bullet_points = createBulletList(bullet_points)

    if(bullet_points == None):
       return jsonify({'transcript': response, 'bulletpoints':["DO NOT GENERATE A QUIZ"]})
    

    # Dummy response for now
    return jsonify({'transcript': response,'bulletpoints':bullet_points})

    


# API endpoint to receive audio file
@app.route('/make_quiz', methods=['POST'])
def quiz():
    data = request.get_json()
    print(data)
    transcript = data['transcript']
    

    # function calls
    bullet_points = text_to_bullet_list(transcript)

    #expensice call i guess
    quiz_prompt = bullet_list_to_quiz(bullet_points)
    final = format_quiz(quiz_prompt)
    quiz_json = {}

    for key,question in final.items():
        quiz_json[key] = {
            'question': question.question,
            'options': question.options,
            'answer': question.answer
        }

    
    return jsonify({'quiz': quiz_json})


if __name__ == '__main__':
    app.run(debug=True)



