import telebot
from summarize import get_final_answers

bot=telebot.TeleBot('7460104082:AAG1Sj0cPUBOCGp2hkHT1wW08AmplnWHinU')

@bot.message_handler(commands=['start'])
def start(message):
    bot.send_message(message.chat.id,"Hi, This is BSK trying out a temporary version of our study chatbot ask a question related to data structures",parse_mode='html')

@bot.message_handler(content_types='text')
def answer(message):
    ans=get_final_answers(message.text)
    if len(ans)>4096:
        ans=ans[:4096]
    bot.reply_to(message,ans)
    bot.send_message(message.chat.id,"Code:",parse_mode='html')


bot.infinity_polling()

