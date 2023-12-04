from content_generation.generate_audio import generate_audio, generate_conversation
from content_generation.generate_script import generate_script
from openai_api.key_validation import retrieve_api_key
from transcription.transcribe_audio import transcribe_audio

from langchain.chat_models import ChatOpenAI


def get_stick_to_script_score(api_key, script, conversation_text):
    score_prompt = f"""Considering this script {script} and this conversation {conversation_text}, 
    expressed in percentages, how often does the salesperson stick to the script, instead of
    deviating? Only reply with the percentage."""

    chat_model: ChatOpenAI = ChatOpenAI(model_name="gpt-3.5-turbo", api_key=api_key)
    return chat_model.predict(score_prompt)


if __name__ == "__main__":
    print("VALIDATING API KEY")
    api_key = retrieve_api_key()

    print("GENERATING SALES SCRIPT")
    script_scenario = "solar roofing door to door sales"
    stick_to_script_ratio = 0.7
    script = generate_script(api_key, script_scenario)
    
    print("GENERATING SALES CONVERSATION AUDIO")
    generate_conversation(api_key, script, stick_to_script_ratio)

    print("TRANSCRIBING SALES CONVERSATION AUDIO")
    conversation_text = transcribe_audio()
    print(get_stick_to_script_score(api_key, script, conversation_text))
