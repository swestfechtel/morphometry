// API base URL used by all client fetches.
//
// Override per deployment with the NEXT_PUBLIC_MODEL_API environment variable
// (e.g. in frontend/.env.local). NEXT_PUBLIC_* values are inlined at build time
// by Next.js, so a rebuild is required after changing it. Defaults to the local
// API so development works out of the box.
const server_config = {
  model_api: process.env.NEXT_PUBLIC_MODEL_API ?? "http://localhost:8000",
};

export default server_config;
