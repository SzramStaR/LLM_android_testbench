export interface Env {
  DEBUG_BUCKET: R2Bucket;
	PROTECT_KEY: string;  
}

const hasValidHeader = (request: Request, env: Env) => {
  return request.headers.get("X-PROTECT-KEY") === env.PROTECT_KEY;
};


export default {
  async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    if (request.method !== 'POST' || !request.url.endsWith('/saveBenchmark') || !hasValidHeader(request, env)) {
      return new Response(null, { status: 418 });
    }

    try {
      const jsonData = await request.json();
      
      const timestamp = String(Date.now());
			const randomNum = Math.floor(Math.random() * 1000);
      const filename = `benchmark-${timestamp}-${randomNum}.json`;
      
      const jsonString = JSON.stringify(jsonData, null, 2);
      
      await env.DEBUG_BUCKET.put(filename, jsonString, {
        customMetadata: {
          'content-type': 'application/json',
          'created-at': timestamp
        }
      });
      
      return new Response(null, { status: 200 });
    } catch (error) {
			console.log('Error:', error);
      return new Response(null, { status: 500 });
    }
  }
};