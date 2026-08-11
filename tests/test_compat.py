from snd import TNFigure, quick_draw


def test_quick_draw_returns_a_drawn_tnfigure(qtn_chain):
    fig = quick_draw(qtn_chain)

    assert isinstance(fig, TNFigure)
    assert fig._drawing is not None  # already drawn
