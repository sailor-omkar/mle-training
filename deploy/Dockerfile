FROM tigeromkar/mle_dev:latest

WORKDIR /app

ADD house_price_prediction_omkar /app/house_price_prediction_omkar/

ENV PYTHONPATH "${PYTHONPATH}:/app"

WORKDIR /app/house_price_prediction_omkar

CMD ["bash","main_script.sh"]

EXPOSE 8080
