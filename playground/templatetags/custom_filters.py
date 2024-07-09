from django import template

register = template.Library()

@register.filter
def format_number(value):
    try:
        if value == None:
            return 'N/A'
        value = float(value)
        if abs(value) >= 1_000_000_000_000:
            return f'{value / 1_000_000_000_000:.1f}T'
        elif abs(value) >= 1_000_000_000:
            return f'{value / 1_000_000_000:.1f}B'
        elif abs(value) >= 1_000_000:
            return f'{value / 1_000_000:.1f}M'
        elif abs(value) >= 1_000:
            return f'{value / 1_000:.1f}K'
        else:
            return str(round(value, 3))
    except (ValueError, TypeError):
        return value