FROM node:20-slim AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
ARG VITE_API_BASE
ARG VITE_DASHBOARD_BASE=/
ENV VITE_API_BASE=$VITE_API_BASE
ENV VITE_DASHBOARD_BASE=$VITE_DASHBOARD_BASE
RUN npm run build

FROM nginx:1.25-alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
