# Django Trading Bot

This project is a web application for a trading bot that displays financial data and indicators. It includes functionality for researching financial data, displaying market indicators, and more.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Running Tests](#running-tests)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The Django Trading Bot is designed to help users analyze financial data and market indicators. It provides a user-friendly interface to view financial metrics, perform technical analysis, and offers additional functionalities like the efficient portfolio with the most recent stock market data and AI news sentiment analysis.

## Features

- Display financial data for selected stocks
- Show technical indicators like SMA, EMA, Bollinger Bands, RSI, MACD, and Stochastic
- Interactive research page to view stock data and charts
- Generate an efficient portfolio with the latest stock market data
- AI-powered news sentiment analysis for informed decision-making
- Admin panel for managing data and settings

## Installation

Follow these steps to set up the project on your local machine.

### Prerequisites

- Python 3.7+
- Django 3.0+
- Virtualenv (optional but recommended)

### Steps

1. **Clone the repository**

    ```sh
    git clone https://github.com/yourusername/django-trading-bot.git
    cd django-trading-bot
    ```

2. **Create and activate a virtual environment**

    ```sh
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

3. **Install the required packages**

    ```sh
    pip install -r requirements.txt
    ```

4. **Apply migrations**

    ```sh
    python manage.py migrate
    ```

5. **Run the development server**

    ```sh
    python manage.py runserver
    ```

6. **Access the application**

    Open your browser and go to `http://127.0.0.1:8000/`.

## Usage

### Admin Panel

- Access the admin panel at `http://127.0.0.1:8000/admin/`.
- Create a superuser to access the admin panel:

    ```sh
    python manage.py createsuperuser
    ```

### Research Page

- Visit `http://127.0.0.1:8000/research/` to use the research functionalities.
- Click on the stock tickers to view detailed financial data and indicators.

### Efficient Portfolio

- Visit `http://127.0.0.1:8000/portfolio/` to view the efficient portfolio generated using the most recent stock market data.

### AI News Sentiment Analysis

- Visit `http://127.0.0.1:8000/news-sentiment/` to see AI-powered news sentiment analysis, helping you make informed trading decisions.

## Configuration

- Update settings in `settings.py` as per your environment and requirements.
- Configure database settings, email settings, and other environment-specific variables.

## Running Tests

- Run tests using the following command:

    ```sh
    python manage.py test
    ```

## Contributing

We welcome contributions! Please read `CONTRIBUTING.md` for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
