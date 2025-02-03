
# using python 3.9 slim image for smaller image size
FROM python:3.9-slim

# now set the working directory in the container for the app
WORKDIR /app

# copying all the files from the current directory to the container working directory
COPY . /app

# install the requirements.txt file
RUN pip install --no-cache-dir -r requirements.txt

# expose the port 8000 so that we can access the FastAPI app
EXPOSE 8000

# command to run the FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]