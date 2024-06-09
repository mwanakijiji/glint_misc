from PIL import Image
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches

stem = '/Users/bandari/Documents/postdoc_sydney/conferences/spie_2024/'

def gen_hr4796a():
    ################################
    # HR4796A debris disk
    ################################
    # Read the JPEG image
    image_path = stem + 'perrin_hr4796A.jpeg'
    image = Image.open(image_path)
    # Create a figure and axes
    fig, ax = plt.subplots()
    # Display the image
    ax.imshow(image)
    # Add a horizontal line to measure scale
    '''
    x = 1036
    y = 517
    length = 92
    line = patches.ConnectionPatch((x, y), (x + length, y), "data", "data", edgecolor='red')
    ax.add_artist(line)
    '''
    pix_per_arcsec = 92./0.5
    # red circle: Jupiter orbit 
    radius_au = 5.2
    d_pc = 71
    radius_asec = radius_au/d_pc
    radius_pix = radius_asec * pix_per_arcsec
    circle_center = (1194, 292.2)
    text_position = (1200, 340)
    #circle = patches.Circle((1193, 292.2), radius_pix, edgecolor='red', facecolor='none')
    circle = patches.Circle(circle_center, radius_pix, edgecolor='red', facecolor='none')
    ax.add_patch(circle)
    # Add a text annotation
    ax.text(1200, 370, '$a_{J}$', ha='center', va='center', color='red', fontsize=14)
    print('------')
    print('MWC 758')
    print('Resolution elements lambda/(2B) in H-band for 8m telescope within a_J:', 2*radius_asec/beam_8m_asec)
    print('Resolution elements lambda/(2B) in H-band for 30m telescope within a_J:', 2*radius_asec/beam_30m_asec)
    # Draw a line connecting the circle to the text
    # Calculate the starting point of the line at the perimeter of the circle
    line_start = (
        circle_center[0] + radius_pix * math.cos(math.radians(80)),
        circle_center[1] + radius_pix * math.sin(math.radians(80))
    )

    # the Kuiper belt
    # Draw a shaded region between the inner and outer radii
    radius_au_in = 30
    radius_au_out = 50
    inner_radius_asec = radius_au_in / d_pc
    outer_radius_asec = radius_au_out / d_pc
    inner_radius_pix = inner_radius_asec * pix_per_arcsec
    outer_radius_pix = outer_radius_asec * pix_per_arcsec
    annulus = patches.Wedge((1194, 292.2), outer_radius_pix, 0, 360, width=outer_radius_pix-inner_radius_pix, facecolor='gray', alpha=0.5)
    ax.add_patch(annulus)

    ax.plot([line_start[0], text_position[0]], [line_start[1], text_position[1]], color='red', linestyle='-', linewidth=0.5)
    #ax.plot([line_start[0], text_position[0]], [line_start[1], text_position[1]], color='red', linestyle='-', linewidth=0.5)

    plt.show()
    #plt.savefig('junk_1.pdf', dpi=300)
    plt.clf()

    return


def gen_mwc758():
    ################################
    # MWC 758 PP disk
    ################################
    # Read the JPEG image
    image_path = stem + 'mwc_758_stsci.jpeg'
    image = Image.open(image_path)
    # Create a figure and axes
    fig, ax = plt.subplots()
    # Display the image
    ax.imshow(image)
    # Add a horizontal line to measure scale
    '''
    x = 366
    y = 930
    length = 265
    line = patches.ConnectionPatch((x, y), (x + length, y), "data", "data", edgecolor='red')
    ax.add_artist(line)
    '''
    
    pix_per_arcsec = 265./0.37
    # red circle: Jupiter orbit
    radius_au = 5.2
    d_pc = 279
    radius_asec = radius_au/d_pc
    radius_pix = radius_asec * pix_per_arcsec
    circle_center = (498, 494)
    text_position = (409, 642)
    #circle = patches.Circle((1193, 292.2), radius_pix, edgecolor='red', facecolor='none')
    circle = patches.Circle(circle_center, radius_pix, edgecolor='red', facecolor='none')
    ax.add_patch(circle)
    # Add a text annotation
    ax.text(text_position[0]-20, text_position[1]+20, '$a_{J}$', ha='center', va='center', color='red', fontsize=14)
    print('------')
    print('HR 4796A')
    print('Resolution elements lambda/(2B) in H-band for 8m telescope within a_J:', 2*radius_asec/beam_8m_asec)
    print('Resolution elements lambda/(2B) in H-band for 30m telescope within a_J:', 2*radius_asec/beam_30m_asec)
    # Draw a line connecting the circle to the text
    # Calculate the starting point of the line at the perimeter of the circle
    line_start = (
        circle_center[0] + radius_pix * math.cos(math.radians(110)),
        circle_center[1] + radius_pix * math.sin(math.radians(110))
    )

    # the Kuiper belt
    # Draw a shaded region between the inner and outer radii
    radius_au_in = 30
    radius_au_out = 50
    inner_radius_asec = radius_au_in / d_pc
    outer_radius_asec = radius_au_out / d_pc
    inner_radius_pix = inner_radius_asec * pix_per_arcsec
    outer_radius_pix = outer_radius_asec * pix_per_arcsec
    annulus = patches.Wedge(circle_center, outer_radius_pix, 0, 360, width=outer_radius_pix-inner_radius_pix, facecolor='gray', alpha=0.5)
    #annulus = patches.Wedge((156,886), outer_radius_pix, 0, 360, width=outer_radius_pix-inner_radius_pix, facecolor='gray', alpha=0.5)

    ax.add_patch(annulus)

    ax.plot([line_start[0], text_position[0]], [line_start[1], text_position[1]], color='red', linestyle='-', linewidth=0.5)
    #ax.plot([line_start[0], text_position[0]], [line_start[1], text_position[1]], color='red', linestyle='-', linewidth=0.5)

    plt.show()
    #plt.savefig('junk_2.pdf', dpi=300)
    plt.clf()

    return 


def gen_hr8799():
    ################################
    # HR 8799 planets
    ################################
    # Read the JPEG image
    image_path = stem + 'currie_hr8799.jpeg'
    image = Image.open(image_path)
    # Create a figure and axes
    fig, ax = plt.subplots()
    # Display the image
    ax.imshow(image)
    # Add a horizontal line to measure scale

    x = 366
    y = 930
    length = 265
    line = patches.ConnectionPatch((x, y), (x + length, y), "data", "data", edgecolor='red')
    ax.add_artist(line)

    
    pix_per_arcsec = 265./0.37
    # red circle: Jupiter orbit
    radius_au = 5.2
    d_pc = 40.8 # pc
    radius_asec = radius_au/d_pc
    radius_pix = radius_asec * pix_per_arcsec
    circle_center = (498, 494)
    text_position = (409, 642)
    #circle = patches.Circle((1193, 292.2), radius_pix, edgecolor='red', facecolor='none')
    circle = patches.Circle(circle_center, radius_pix, edgecolor='red', facecolor='none')
    ax.add_patch(circle)
    # Add a text annotation
    ax.text(text_position[0]-20, text_position[1]+20, '$a_{J}$', ha='center', va='center', color='red', fontsize=14)
    print('------')
    print('HR 4796A')
    print('Resolution elements lambda/(2B) in H-band for 8m telescope within a_J:', 2*radius_asec/beam_8m_asec)
    print('Resolution elements lambda/(2B) in H-band for 30m telescope within a_J:', 2*radius_asec/beam_30m_asec)
    # Draw a line connecting the circle to the text
    # Calculate the starting point of the line at the perimeter of the circle
    line_start = (
        circle_center[0] + radius_pix * math.cos(math.radians(110)),
        circle_center[1] + radius_pix * math.sin(math.radians(110))
    )

    # the Kuiper belt
    # Draw a shaded region between the inner and outer radii
    radius_au_in = 30
    radius_au_out = 50
    inner_radius_asec = radius_au_in / d_pc
    outer_radius_asec = radius_au_out / d_pc
    inner_radius_pix = inner_radius_asec * pix_per_arcsec
    outer_radius_pix = outer_radius_asec * pix_per_arcsec
    annulus = patches.Wedge(circle_center, outer_radius_pix, 0, 360, width=outer_radius_pix-inner_radius_pix, facecolor='gray', alpha=0.5)
    #annulus = patches.Wedge((156,886), outer_radius_pix, 0, 360, width=outer_radius_pix-inner_radius_pix, facecolor='gray', alpha=0.5)

    ax.add_patch(annulus)

    ax.plot([line_start[0], text_position[0]], [line_start[1], text_position[1]], color='red', linestyle='-', linewidth=0.5)
    #ax.plot([line_start[0], text_position[0]], [line_start[1], text_position[1]], color='red', linestyle='-', linewidth=0.5)

    plt.show()
    #plt.savefig('junk_2.pdf', dpi=300)
    plt.clf()

    return 


if __name__ == "__main__":

    # lambda/2B for H-band, 8-m
    beam_8m_asec = ((1.6e-6)/8)*206265
    beam_30m_asec = ((1.6e-6)/30)*206265

    _ = gen_hr4796a()
    _ = gen_mwc758()
    _ = gen_hr8799()


