# syntax=docker/dockerfile:1
# Frontend image: Next.js standalone server. Build context is frontend/.
#   docker build --build-arg MORPH_VERSION=1.2.3 -t swestfechtel/morphometry_frontend:1.2.3 frontend
# Runtime: -e MODEL_API_INTERNAL=http://<api-host>:8000 (server-side rendering fetches);
# the browser derives the API URL from the page hostname (see app/server_config.ts).

FROM node:22.14.0-bookworm-slim AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

FROM node:22.14.0-bookworm-slim AS build
WORKDIR /app
ENV NEXT_TELEMETRY_DISABLED=1
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN npm run build

FROM node:22.14.0-bookworm-slim AS runtime
ARG MORPH_VERSION=dev
LABEL org.opencontainers.image.version=$MORPH_VERSION
ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    NEXT_PUBLIC_APP_VERSION=$MORPH_VERSION \
    PORT=3000 \
    HOSTNAME=0.0.0.0
WORKDIR /app
COPY --from=build /app/.next/standalone ./
COPY --from=build /app/.next/static ./.next/static
COPY --from=build /app/public ./public
USER node
EXPOSE 3000
CMD ["node", "server.js"]
