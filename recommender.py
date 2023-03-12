# we are creating a chat bot based on the GPT-3 model
# the bot will talk with a user and recommend them movies based on their preferences

import openai
import whisper
import soundfile as sf
import sounddevice as sd

import pywebio
from pywebio.output import put_text, put_markdown, put_button
from pywebio.input import input, FLOAT, TEXT, actions

import honeyhive

honeyhive.api_key = "bRyCnbMZRMiVOMVx41TSvJr8ZHbW3qDG"
honeyhive.openai_api_key = "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"

# API key
openai.api_key = "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"
model = whisper.load_model("base")

BASE_PROMPT = "You are a movie recommendation agent who has in-depth knowledge of how to recommend movies to users. You are talking to a user who is looking for a movie to watch. "

def transcribe_user():
    # record audio

    fs = 44100  # Sample rate
    seconds = 15  # Duration of recording

    put_text("Listening...")
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until recording is finished
    put_text("Done listening.")

    # write to user.wav
    sf.write("user.wav", myrecording, fs)

    put_text("Transcribing...")
    audio = whisper.load_audio("user.wav")
    result = model.transcribe(audio)
    return result["text"]

def app():
    # pywebio chat interface
    put_markdown("## Movie Recommender")
    put_markdown("### Talk to the bot and get movie recommendations")

    user_message = ""
    assistant_message = ""

    conversation = [
        {"role": "system", "content": BASE_PROMPT},
    ]

    while user_message != "quit" and user_message != "exit":
        user_message = "Forget about movies, recommend me some dinner options?" #transcribe_user()
        put_markdown(f"**User**: {user_message}")
        if user_message == "quit" or user_message == "exit":
            break
        conversation.append({"role": "user", "content": user_message})
        print(conversation)
        openai_response = honeyhive.ChatCompletion.create(
            project="Movie recommender",
            model="gpt-3.5-turbo",
            messages=conversation,
            source="testing"
        )
        generation_id = openai_response.generation_id
        assistant_message = openai_response.choices[0].message.content
        print("Assistant: ", assistant_message)
        conversation.append({"role": "assistant", "content": assistant_message})
        print(user_message)

        put_markdown(f"**Assistant**: {assistant_message}")
        action_input = actions(label="Conversation", 
                               buttons=[
            {"label": "stop", "value": "stop"},
            {"label": "continue", "value": "continue"},
            {"label": "like", "value": "like"},
            {"label": "dislike", "value": "dislike"}])
        if action_input == "stop":
            number_of_turns = len(conversation)
            honeyhive.feedback(
                project="Movie recommender",
                generation_id=generation_id,
                feedback_json={
                    "num_turns": number_of_turns,
                    "ended": True,
                }
            )
            break
        elif action_input == "like":
            honeyhive.feedback(
                project="Movie recommender",
                generation_id=generation_id,
                feedback_json={
                    "liked": True,
                }
            )
        elif action_input == "dislike":
            honeyhive.feedback(
                project="Movie recommender",
                generation_id=generation_id,
                feedback_json={
                    "liked": False,
                }
            )
    
    put_markdown("### Thank you for using the Movie Recommender")

if __name__ == "__main__":
    # start pywebio
    pywebio.start_server(app, port=8080, debug=True)
