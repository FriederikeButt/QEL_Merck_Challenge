# Stage 1: Compile and Build angular codebase

# Use official node image as the base image
FROM node:latest as build-step

# Create app directory
RUN mkdir -p /app

# Change workdir
WORKDIR /app

# Copy everything
COPY . /app

# Install all the dependencies
RUN npm install

# Generate the build of the application
RUN npm run build 

FROM nginx:1.20.1

COPY --from=build-step app/dist/qtex /usr/share/nginx/html


EXPOSE 4200:80
