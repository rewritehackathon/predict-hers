const functions = require('firebase-functions');
const {WebhookClient} = require('dialogflow-fulfillment');
const {Card, Suggestion} = require('dialogflow-fulfillment');

process.env.DEBUG = 'dialogflow:debug'; // enables lib debugging statements

exports.dialogflowFirebaseFulfillment = functions.https.onRequest((request, response) => {
  const agent = new WebhookClient({ request, response });
  
  function equipfun(agent) {
    
    let params = agent.getContext('location-followup').parameters;
    let state = params.param1;

    //console.log(agent.context.get('location-followup').params.state);
    agent.add('These are the insurances you need. Workers compensation insurance,  General liability insurance, Property insurance if not online business,       Business interruption. For automobile you can have Product liability insurance, Disability insurance, pollution insurance. In your region  you should have commercial flood insurance,  Earthquake insurance,  Terrorism insurance,   Political risk insurance');
  }

  // Run the proper function handler based on the matched Dialogflow intent name
  let intentMap = new Map();
  intentMap.set('equipment', equipfun);
  agent.handleRequest(intentMap);
});