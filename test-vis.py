
import visvis2 as vv


f = vv.QtFigure()

v = vv.View()
f._views.append(v)  # todo: API?

t = vv.Triangle()

v.scene.children.append(t)  # todo: API?



r = vv.Renderer()
# r._surface_id = f._get_surface_id(r._ctx)  # todo: API? only needed in _create_swapchain
# r._create_swapchain()

r.collect_from_figure(f)


async def drawer():
    while True:
        await asyncio.sleep(0.1)
        # print("draw")
        r.draw_frame(f)


import asyncio
asyncio.get_event_loop().create_task(drawer())

if __name__ == "__main__":
    import asyncio

    loop = asyncio.get_event_loop()
    if hasattr(asyncio, "integrate_with_ide"):
        asyncio.integrate_with_ide(loop, run=False)
    loop.run_forever()
