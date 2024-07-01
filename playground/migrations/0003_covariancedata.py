# Generated by Django 5.0.6 on 2024-07-01 04:03

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('playground', '0002_remove_stockdata_high_price_and_more'),
    ]

    operations = [
        migrations.CreateModel(
            name='CovarianceData',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('tickers', models.CharField(max_length=255)),
                ('covariance_matrix', models.TextField()),
                ('created_at', models.DateTimeField(auto_now_add=True)),
            ],
        ),
    ]
