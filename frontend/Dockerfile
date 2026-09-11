# SynapseFly frontend (the visible page). Build context = repo root:
#   docker build -f frontend/Dockerfile -t synapsefly-frontend \
#     --build-arg NEXT_PUBLIC_WS_URL=wss://api.synapsefly.com/ws \
#     --build-arg NEXT_PUBLIC_API_URL=https://api.synapsefly.com .
# NEXT_PUBLIC_* are inlined at build time, so they are passed as build args.

FROM node:20-slim AS build
WORKDIR /app
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
ARG NEXT_PUBLIC_WS_URL
ARG NEXT_PUBLIC_API_URL
ENV NEXT_PUBLIC_WS_URL=$NEXT_PUBLIC_WS_URL \
    NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL \
    NEXT_TELEMETRY_DISABLED=1
RUN npm run build

FROM node:20-slim AS run
WORKDIR /app
ENV NODE_ENV=production \
    PORT=3000 \
    HOSTNAME=0.0.0.0 \
    NEXT_TELEMETRY_DISABLED=1
# Standalone output already contains the minimal node_modules it needs.
COPY --from=build /app/.next/standalone ./
COPY --from=build /app/.next/static ./.next/static
COPY --from=build /app/public ./public
EXPOSE 3000
CMD ["node", "server.js"]
