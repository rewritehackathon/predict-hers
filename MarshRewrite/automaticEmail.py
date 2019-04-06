
import smtplib

from string import Template

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

MY_ADDRESS = 'predicthers@gmail.com'
PASSWORD = 'qawsedrftgyhujikol'

def getContacts(filename):
    """
	Store all email address and people name in the txt file
    """
    
    names = []
    emails = []
    with open(filename, mode='r', encoding='utf-8') as contacts_file:
        for a_contact in contacts_file:
            names.append(a_contact.split()[0])
            emails.append(a_contact.split()[1])
    return names, emails

def readTemplate(filename):
    """
   Create the template 
    """
    
    with open(filename, 'r', encoding='utf-8') as templateFile:
        templateContent = templateFile.read()
    return Template(templateContent)

def main():
    names, emails = getContacts('contacts.txt') # read contacts
    messageTemplate = readTemplate('messageBody.txt')

    # set up the SMTP server

    s = smtplib.SMTP(host='smtp.gmail.com', port=587)
    s.starttls()
    s.login(MY_ADDRESS, PASSWORD)

    for name, email in zip(names, emails):
    	# create a message
        msg = MIMEMultipart()  

        # add in the actual person name to the message template
        message = messageTemplate.substitute(PERSON_NAME=name.title())
        print(message)
        # setup the parameters of the message
        msg['From']=MY_ADDRESS
        msg['To']=email
        msg['Subject']="ALERT ALERT ALERT !!!!!!!"
        
        # add in the message body
        msg.attach(MIMEText(message, 'plain'))
        
        # send the message via the server set up earlier.
        s.send_message(msg)
        del msg
        
    # Terminate the SMTP session and close the connection
    s.quit()
    
if __name__ == '__main__':
    main()