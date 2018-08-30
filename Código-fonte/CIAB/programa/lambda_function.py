import os
import boto3

bucketName = 'ciab2018-bike-iot'
fileHeader = 'drone-'
fileType = '.txt'
SECRET_KEY = 'dr0neRulez2A5T7U'

def lambda_handler(event, context):
"""
    Código utilizado no AWS Lambda, para responder ao site da CIAB
    armazenando ou solicitando o comando para o miniDrone
"""
    content = "Failed"
    secret = event['queryStringParameters']['secret']
    
    # Verifica se o segredo esta correto.
    if SECRET_KEY != secret:
        return formatarRetorno(500, "Segredo Incorreto")

    operation = event['queryStringParameters']['operation']
    droneId = event['queryStringParameters']['droneId']

    fileName = fileHeader + droneId + fileType
    s3 = boto3.client('s3')

    if (operation == "POP" or operation == "PEEK"):
        try:
            response = s3.get_object(Bucket=bucketName, Key=fileName)
            content = response['Body'].read().decode('utf-8')
            if (operation == "POP"):
                s3.delete_object(Bucket=bucketName, Key=fileName)
        except:
            content = 'empty'


    elif (operation == "PUSH"):
        if not ('command' in event['queryStringParameters']):
            return formatarRetorno(500, "Faltou command.")
            
        command = event['queryStringParameters']['command']
        s3.put_object(Bucket=bucketName, Key=fileName, Body=command.encode())
        content = "OK"
        
    print ('Conteudo retornado: %s' % content)
    return formatarRetorno(200, content)
    

def formatarRetorno(statusCode, content, contentType = "text/plain"):
    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": contentType,
            "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
            "Access-Control-Allow-Methods": "GET",
            "Access-Control-Allow-Origin": "*"
        },
        "body": content,
        "isBase64Encoded": "false"
    }
