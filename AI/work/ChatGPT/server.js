// ref: https://deno.land/x/openai@1.3.4
import { OpenAI } from "https://deno.land/x/openai/mod.ts";
import { Application, Router, send } from "https://deno.land/x/oak/mod.ts";

const openAI = new OpenAI("Your API Key!")

const router = new Router(); 

router.post('/getgpt', getgpt)
      .get("/(.*)", async (ctx) => {  // put file to web
        await send(ctx, ctx.params[0],{
          root: Deno.cwd(),
          index: "index.html"
        })
      })
      

const app = new Application();

app.use(router.routes())
app.use(router.allowedMethods())

// read fetch body and response gpt text
async function getgpt (ctx) {
	  const body = ctx.request.body();
	  console.log('body = ', body)
    let post = await body.value;
    console.log(post)

    const chatCompletion = await openAI.createChatCompletion({
      model: "gpt-3.5-turbo",
      messages: [
        { "role": "system", "content": "You are a helpful assistant.Using traditional Chinese." },
        { "role": "user", "content": post.content },
      ],
    });
    
    console.log(chatCompletion.choices[0].message.content);
	  
	  ctx.response.type = 'application/json'
    ctx.response.body = { "get_res": chatCompletion.choices[0].message.content }
}


console.log('start at : http://127.0.0.1:8000')  // run on port 8000
console.log('cwd=', Deno.cwd())
await app.listen({ port: 8000 });