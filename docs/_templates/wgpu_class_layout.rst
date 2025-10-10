{{ objname | escape | underline}}

.. currentmodule:: {{ module }}

{% if objname == "GPUPromise" %}

.. autoclass:: {{ objname }}
    :members:
    :inherited-members:
    :show-inheritance:

{% else %}

.. autoclass:: {{ objname }}
    :members:
    :show-inheritance:

{% endif %}
