"""Backward-compatible, single-call entry point mirroring quimb's own
tn.draw(tn, **kwargs) shape, for quick figures and easy migration from
the old figures.org scripts."""

from .figure import TNFigure


def quick_draw(tn, layout: str = "auto", figsize=(6, 4), **style_kwargs) -> TNFigure:
    fig = TNFigure(tn, figsize=figsize).layout(kind=layout)
    for tag, kwargs in style_kwargs.items():
        fig.style(tag, **kwargs)
    return fig.draw()
