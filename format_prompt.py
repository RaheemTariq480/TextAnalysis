from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain.prompts import ChatPromptTemplate


def get_formatted_prompt(text_, category_list):
    emotion_schema = ResponseSchema(name="emotion",
                                description="Does the text expresses any emotion? If yes, respond with the emotion, such as joy, anger, sadness, or fear.; if no, respond with 'neutral'.")
    sentiment_schema = ResponseSchema(name="sentiment",
                                        description="Does the text expresses any sentiment? If yes, respond with the emotion;  if no, respond with 'neutral'.")
    category_schema = ResponseSchema(name="category",
                                        description="categorize the text in the respective category from the list")

    response_schemas = [emotion_schema, sentiment_schema, category_schema]

    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    prompt_start = """\
    For the following text, extract the following information:

    emotion: Does the text expresses any emotion?
    such as joy, anger, sadness, or fear.
    If yes, respond with the emotion;
    if no, respond with 'neutral'.

    sentiment: Does the text expresses any positive, negative or neutral sentiment?
    If yes, respond with the sentiment;
    if no, respond with 'neutral'.

    category: categorize the text in the respective category from the list
    category_list : {categorylist}
    text: {text}

    {format_instructions}
    """

    prompt = ChatPromptTemplate.from_template(template=prompt_start)
    messages = prompt.format_messages(text=text_,
                                    format_instructions=format_instructions,
                                    categorylist=category_list)
    return  messages[0].content
