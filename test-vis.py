
import visvis2 as vv


f = vv.Figure()

v = vv.View()
f._views.append(v)  # todo: API?

t = vv.Triangle()

v.scene.children.append(t)  # todo: API?


if __name__ == "__main__":
    import asyncio

    loop = asyncio.get_event_loop()
    loop.run_forever()
