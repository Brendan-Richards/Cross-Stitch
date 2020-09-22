import CrossStitch
import imageio
import matplotlib.pyplot as plt

def test_needs_stitch():
    all_black = imageio.imread('./images/test/all_black.jpg')
    assert(CrossStitch.needs_stitch(all_black, 0, 0, 15, 15) == False)

    slash = imageio.imread('./images/test/slash.jpg')
    assert(CrossStitch.needs_stitch(slash, 0, 0, 15, 15) == True)

    backslash = imageio.imread('./images/test/backslash.jpg')
    assert(CrossStitch.needs_stitch(backslash, 0, 0, 15, 15) == True)

    bottom_line = imageio.imread('./images/test/bottom_line.jpg')
    assert(CrossStitch.needs_stitch(bottom_line, 0, 0, 15, 15) == True)

    top_line = imageio.imread('./images/test/top_line.jpg')
    assert(CrossStitch.needs_stitch(top_line, 0, 0, 15, 15) == True)


def test_get_backstitch():
    all_black = imageio.imread('./images/test/all_black.jpg')
    assert(CrossStitch.get_backstitch(all_black, 0, 0, 15, 15) == None)

    backslash_small = imageio.imread('./images/test/backslash_small.jpg')
    assert (CrossStitch.get_backstitch(backslash_small, 0, 0, 5, 5) == 'backslash')

    slash = imageio.imread('./images/test/slash.jpg')
    assert (CrossStitch.get_backstitch(slash, 0, 0, 15, 15) == 'slash')

    backslash = imageio.imread('./images/test/backslash.jpg')
    assert (CrossStitch.get_backstitch(backslash, 0, 0, 15, 15) == 'backslash')

    bottom_line = imageio.imread('./images/test/bottom_line.jpg')
    assert (CrossStitch.get_backstitch(bottom_line, 0, 0, 15, 15) == 'bottom')

    top_line = imageio.imread('./images/test/top_line.jpg')
    assert (CrossStitch.get_backstitch(top_line, 0, 0, 15, 15) == 'top')

    left_line = imageio.imread('./images/test/left_line.jpg')
    assert (CrossStitch.get_backstitch(left_line, 0, 0, 15, 15) == 'left')

    right_line = imageio.imread('./images/test/right_line.jpg')
    assert (CrossStitch.get_backstitch(right_line, 0, 0, 15, 15) == 'right')


def test_set_backstitch():
    right_line = imageio.imread('./images/test/bottom_line.jpg')
    CrossStitch.set_backstitch(right_line, 0, 0, 15, 15, 'backslash')
    plt.imshow(right_line)
    plt.show()


if __name__=='__main__':
    test_needs_stitch()
    test_get_backstitch()
    test_set_backstitch()