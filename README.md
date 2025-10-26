
# GSR-Based Polygraph Experimental Pipeline

This project provides a data analysis pipeline for a (self-built) polygraph. A measurement node sends the data from a GSR sensor under various pressure and humidity conditions to the central server with a web interface for control, which automatically analyzes it.


## How to use

- Create an account and log in.
- Create a new API key for your measurement nodes.

- Start your node and enter the API key.

- Change the node configuration in the web UI to the current pressure and humidity values.

- Record.
- Create baseline values.

- Change the node configuration for each measurement series.

- Generate the PDF evaluation.


## API Reference

#### Get Node Config

```http
  GET /api/config
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `token` | `string` | **Required**. Your API key |

#### Upload Measurment Data

```http
  GET /api/upload
```

| token | device_id     | gsr | pulse | humidity | pressure | metadata |
| :-------- | :------- |  :-------- | :-------- | :------- |  :-------- |  :-------- |
| `API Key`      | `string` | `float` | `float` | `float` | `float` | `JSON` |




## Demo

[polygraph-w.ddns.net](https://polygraph-w.ddns.net)


## Install

Clone the project

```bash
  git clone https://github.com/Moritydeinyer/polygraph-w-seminar
```

Go to the project directory

```bash
  cd polygraph-w-seminar
```

Start streamlit server

```bash
  docker-compose up --build -d
```


