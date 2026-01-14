import { openai, supabase } from './config.js';
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

const query = "movie with top-secret Manhattan Project";
// Use OpenAI to make the response conversational
const chatMessages = [{
    role: 'system',
    content: `You are an enthusiastic movies expert who loves recommending movies to people. You will be given two pieces of information, some context about podcasts episodes and a question. Your main job is to formulate a short answer to the question using the provided context. If you are unsure and cannot find the answer in the context, say, "Sorry, I don't know the answer." Please do not make up the answer.` 
}];

/* Split movies.txt into text chunks.
Return LangChain's "output" – the array of Document objects. */
async function splitDocument(document) {
 const response = await fetch(document);
  const text = await response.text();

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 150,
    chunkOverlap: 15,
  });
  const output = await splitter.createDocuments([text]);
  console.log((output[0].pageContent));
  return output;
}

/* Create an embedding from each text chunk.
Store all embeddings and corresponding text in Supabase. */
async function createAndStoreEmbeddings() {
  const chunkData = await splitDocument("movies.txt");
  const data = await Promise.all(
    chunkData.map( async (textChunk) => {
        const embeddingResponse = await createEmbedding(textChunk.pageContent)
        return { 
          content: textChunk.pageContent, 
          embedding: embeddingResponse
        }
    })
  );
  // console.log('createAndStoreEmbeddings', data)
  // Insert content and embedding into Supabase
  await supabase.from('documents').insert(data); 
  console.log('Embedding and storing complete!');
  
}

// Query Supabase and return a semantically matching text chunk
async function findNearestMatch(embedding) {
  const { data } = await supabase.rpc('match_documents', {
    query_embedding: embedding,
    match_threshold: 0.50,
    match_count: 10
  });
  console.log('findNearestMatch', data)
  return JSON.stringify(data);
}

// Create an embedding vector representing the input text
async function createEmbedding(input) {
  const embeddingResponse = await openai.embeddings.create({
    model: "text-embedding-ada-002",
    input
  });
  // console.log('embeddingResponse', embeddingResponse.data[0].embedding)
  return embeddingResponse.data[0].embedding;
}

async function getChatCompletion(text, query) {
  chatMessages.push({
    role: 'user',
    content: `Context: ${text} Question: ${query}`
  });
  
  console.log('chatMessages', chatMessages)
  
  const response = await openai.chat.completions.create({
    model: 'gpt-4',
    messages: chatMessages,
    temperature: 0.5,
    frequency_penalty: 0.5
  });

  console.log(response.choices[0].message.content);
}

async function main(input) {
  // createAndStoreEmbeddings();
  const embedding = await createEmbedding(input);
  const match = await findNearestMatch(embedding);
  await getChatCompletion(match, input);
}

main(query);
