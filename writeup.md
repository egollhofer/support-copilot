For the API exercise I built a conversational support assistant called a Support Copilot. It generates email responses to customers who have emailed the help center of Olivetto, an olive oil company. The repository contains a folder of example customer emails as well as a small knowledge base of documents about the company's policies.

The workflow follows the following steps:
1. A customer sends an email   
2. First LLM call to OpenAI drafts an initial response, using the knowledge base for grounded responses. This returns a structured output containing the email draft, any clarifying questions for the customer, and all relevant details from the knowledge based used in generating the response.
3. Second LLM call to Anthropic for review and critique. It checks the draft email for any hallucinations, poor grounding, or tone issues (in this example too cold, too verbose, overly apologetic). It outputs structured feedback and a verdict regarding the email - revise, needs human, OK.
4. Final LLM call, again to OpenAI, writes a final customer-ready email. It incorporates the original draft and reviewer feedback, and only returns the text of the email.

Because embeddings aren't supported in the API yet I passed the entire knowledge base as a long context rather than retrieving small chunks.

The API made it easy to swap out LLM providers and to change parameters such as model and temperature.