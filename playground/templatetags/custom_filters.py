from django import template

register = template.Library()

@register.filter
def format_number(value):
    try:
        value = float(value)
        if value >= 1_000_000_000_000:
            return f'{value / 1_000_000_000_000:.1f}T'
        elif value >= 1_000_000_000:
            return f'{value / 1_000_000_000:.1f}B'
        elif value >= 1_000_000:
            return f'{value / 1_000_000:.1f}M'
        elif value >= 1_000:
            return f'{value / 1_000:.1f}K'
        else:
            return str(value)
    except (ValueError, TypeError):
        return value