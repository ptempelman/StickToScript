import os
import re
import os.path as osp

import openai

from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from pydub import AudioSegment


def generate_conversation(api_key, script, stick_to_script_ratio):
    conversation_memory_rep: ConversationBufferMemory = ConversationBufferMemory()
    conversation_rep: ConversationChain = ConversationChain(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", api_key=api_key),
        memory=conversation_memory_rep,
    )

    conversation_memory_client: ConversationBufferMemory = ConversationBufferMemory()
    conversation_client: ConversationChain = ConversationChain(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", api_key=api_key),
        memory=conversation_memory_client,
    )

    start_prompt_rep = f"""You are a salesperson, you are starting a conversation with a potential 
        client. Try to convince them to buy from you until they say yes or no. Stick to this script
        {script} and deviate from the script {1-stick_to_script_ratio}% of the time. What is your
        opening? Keep it short!"""
    message_rep = conversation_rep.predict(input=start_prompt_rep)

    start_prompt_client = f"""You are acting as a potential client for a salesperson. You can
        decide yourself whether you want to buy from the salesperson or not. The conversation will
        now start and the salesperson says: {message_rep}, what is your response? Keep it short!
        """
    message_client = conversation_client.predict(input=start_prompt_client)

    conversation_length = 0
    while True:
        print(
            f"\033[94mRep: {message_rep} \033[0m\n\033[33mClient: {message_client}\033[0m"
        )
        generate_audio(api_key, message_rep, f"rep{conversation_length}")
        generate_audio(api_key, message_client, f"client{conversation_length}")

        if (
            conversation_length > 10
            or "let's do it" in message_client.lower()
            or "no thank you" in message_client.lower()
        ):
            break

        message_client_padded = f"""Respond to {message_client}, keep it short!"""
        message_rep = conversation_rep.predict(input=message_client_padded)
        message_rep_padded = f"""Salesperson responded: {message_rep}, what is your response? 
            And keep it short!"""
        if conversation_length > 3:
            message_rep_padded += """Remember: whenever you are convinced, specifically say 'Let's 
                do it!', if you're not, say 'No thank you.'"""
        message_client = conversation_client.predict(input=message_rep_padded)

        conversation_length += 1
    chain_audio_files()


def generate_audio(api_key, text_to_audio, filename_counter):
    client = openai.OpenAI(api_key=api_key)

    content_dir = osp.join(osp.dirname(__file__), ".content")

    if not osp.exists(content_dir):
        os.makedirs(content_dir)

    speech_file_path = osp.join(content_dir, f"{filename_counter}.mp3")

    voice = "alloy" if filename_counter.startswith("rep") else "nova"
    response = client.audio.speech.create(
        model="tts-1",
        voice=voice,
        input=text_to_audio,
    )

    response.stream_to_file(speech_file_path)


def chain_audio_files():
    def sort_key(filename):
        match = re.match(r"(rep|client)(\d+)\.mp3", filename)
        if match:
            prefix, number = match.groups()
            priority = 0 if prefix == 'rep' else 1
            return (int(number), priority)
        return (float('inf'), 0)

    directory = osp.join(osp.dirname(__file__), ".content")

    audio_segments = []

    mp3_files = [f for f in os.listdir(directory) if f.endswith(".mp3")]
    mp3_files.sort(key=sort_key)

    for filename in mp3_files:
        file_path = os.path.join(directory, filename)
        segment = AudioSegment.from_mp3(file_path)
        audio_segments.append(segment)

    combined = sum(audio_segments, AudioSegment.silent(duration=0))

    output_file = os.path.join(directory, "full_conversation.mp3")
    combined.export(output_file, format="mp3")

    for filename in mp3_files:
        file_path = os.path.join(directory, filename)
        if file_path != output_file:
            os.remove(file_path)
